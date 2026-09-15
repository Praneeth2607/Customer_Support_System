"""
Step 7: AI Support Agent

Pipeline: customer message -> TF-IDF retrieval over historical conversations
-> single Gemini call that classifies intent, decides AUTO-HANDLE vs
ESCALATE (with a reason), and drafts a grounded reply -> cached result.

Design decisions (see docs/DECISION_LOG.md for the full writeup):

1. Retrieval, not the LLM, supplies "historical grounding". A local TF-IDF
   cosine-similarity search over the leakage-safe corpus (same protection as
   Step 6 -- see src/classification/leakage_utils.py) finds the top-k most
   similar past (customer message, real Uber reply) pairs. This costs
   nothing per call and keeps the LLM from inventing brand-specific facts
   it wasn't given -- Section 17 of the assignment explicitly requires this.

2. ONE LLM call per message does classification + escalation + drafting
   together, instead of three separate calls or a separate classical
   classifier feeding the LLM. This is cheaper (one call, not several) and
   more coherent (the escalation decision and the draft both come from the
   same reasoning about the same retrieved evidence, rather than a rule
   layer second-guessing a separate classifier's output). Step 6's
   TF-IDF+LogReg baseline is NOT reused here on purpose -- it's the
   comparison point, not a component of the system being compared.

3. driver_conduct_and_safety is defensively force-escalated in code even
   though the prompt states it as a hard rule -- never trust a prompt alone
   to enforce a safety-critical policy.

4. Real historical Uber replies in this dataset are mostly short redirects
   to a help link or a DM request (see Step 3's 63% help-URL rate / 37.6%
   DM-deflection rate). The prompt explicitly tells the model that mirroring
   this generic pattern is CORRECT when the evidence is generic -- inventing
   specificity the real data doesn't have would violate the no-fabrication
   rule, not fix a weakness.

5. Results are cached to disk keyed by golden_id (data/results/
   ai_agent_cache.json), so re-running this script after the first pass
   costs nothing and finishes in seconds -- required for the <15-minute
   reproducibility target once results exist.

6. Provider: Google Gemini (`gemini-3.6-flash`, free-tier `GOOGLE_API_KEY`),
   not Anthropic Claude -- an explicit cost-driven choice made by the project
   owner over the paid Anthropic API. See Decision 13 in docs/DECISION_LOG.md.
   `thinking_level="low"` is used since this is a bounded classification +
   short-drafting task, not open-ended reasoning -- it cut token usage by
   ~3.6x in testing (310 -> 86 total tokens per call) with no quality loss
   observed on spot checks, which matters for free-tier rate/quota limits.
"""

import json
import os
import time
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.classification.leakage_utils import iter_leakage_safe_conversations
from src.classification.taxonomy import INTENTS, INTENT_DEFINITIONS, ESCALATION_POLICIES

load_dotenv()

GOLDEN_SET_PATH = "data/golden/golden_evaluation_set.json"
CACHE_PATH = "data/results/ai_agent_cache.json"
RESULTS_PATH = "data/results/ai_agent_results.json"
MODEL = "gemini-flash-lite-latest"
TOP_K_RETRIEVAL = 3
MAX_RETRIES = 8
# Discovered live (not documented anywhere we could find in advance):
# gemini-3.6-flash on this free-tier key is capped at 20 requests/DAY, far
# too few for a 200-example run. Switched to the lite tier, which carries a
# much higher free daily quota. Pacing stays conservative in case a per-minute
# limit also applies to this model.
CALL_PACING_SECONDS = 5.0


class AgentDecision(BaseModel):
    intent: Literal[
        "cancellation_issue", "fare_and_payment_dispute", "lost_item",
        "driver_conduct_and_safety", "pickup_and_route_issue",
        "account_and_promo_issue", "out_of_scope_or_unclear",
    ]
    intent_confidence: float
    should_escalate: bool
    escalation_reason: str  # required in BOTH directions: why it escalates, or why it's safe to auto-handle
    draft_response: str


def build_retrieval_corpus():
    """TF-IDF index over the leakage-safe corpus, keeping each conversation's
    real first Uber reply as the grounding evidence for retrieval."""
    texts, responses, conv_ids = [], [], []
    for conv in iter_leakage_safe_conversations():
        response = conv.get("first_brand_response_clean")
        if not response:
            continue
        texts.append(conv["first_customer_query_clean"].strip())
        responses.append(response)
        conv_ids.append(conv["conversation_id"])

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english",
        min_df=5, max_df=0.6,
    )
    X = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, "X": X, "texts": texts,
            "responses": responses, "conv_ids": conv_ids}


def retrieve(corpus, message, k=TOP_K_RETRIEVAL):
    q_vec = corpus["vectorizer"].transform([message])
    sims = cosine_similarity(q_vec, corpus["X"]).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return [
        {
            "similarity": round(float(sims[idx]), 4),
            "historical_customer_message": corpus["texts"][idx],
            "historical_uber_response": corpus["responses"][idx],
        }
        for idx in top_idx
    ]


def build_system_prompt():
    taxonomy_lines = []
    for intent in INTENTS:
        d = INTENT_DEFINITIONS[intent]
        p = ESCALATION_POLICIES[intent]
        line = (
            f"- `{intent}` ({d['name']}): {d['definition']} "
            f"Default policy: {p['action']} -- {p['policy_rationale']}"
        )
        for note in d.get("boundary_notes", []):
            line += f"\n  BOUNDARY RULE: {note}"
        taxonomy_lines.append(line)
    taxonomy_text = "\n".join(taxonomy_lines)

    return f"""You are an AI support-triage assistant for Uber's ride-hailing customer support team on Twitter.

For each customer message you must:

1. Classify it into exactly ONE of these 7 intents:
{taxonomy_text}

2. Decide AUTO-HANDLE or ESCALATE to a human agent.

   WHAT "AUTO-HANDLE" MEANS HERE: giving the customer the same kind of short, standard redirect that real Uber support actually gives for this intent (a self-serve link, or a request to DM account details) -- exactly like the RETRIEVED HISTORICAL EVIDENCE shows. This IS a complete, correct auto-handle outcome, even though a human will eventually process the underlying case through Uber's normal backend workflow. Auto-handling does NOT mean you personally resolved the issue or looked anything up -- it means no PRIORITY human specialist review is needed beyond the standard flow. Do not escalate just because a case will eventually need someone to check an account or process a refund -- that is normal and expected for the vast majority of AUTO-HANDLE cases too, including real disputes, real complaints, and real refund requests.

   ESCALATE only when at least one of these specific triggers applies -- do not escalate for vague "this needs human judgment" reasoning alone:
   a. `driver_conduct_and_safety` intent -- ALWAYS, no exceptions, regardless of severity (HARD RULE).
   b. `out_of_scope_or_unclear` about Uber Eats or another business line -- ALWAYS, no ride-support policy applies (HARD RULE). Exception: a harmless, fully-answerable, low-stakes message with no real problem (a joke, a pure feature suggestion, a safely-answerable informational question) may still be auto-handled.
   c. Explicit fraud / unauthorized-charge language -- e.g. "charged for a ride I never took", "hacked", "unauthorized", account takeover.
   d. Repeated or systemic pattern language -- explicit counts or durations like "three times", "every time", "for 5 months", "again", "multiple times", "highlighted this before".
   e. The customer explicitly asks for a human / phone call, or demands an action only a human can take (e.g. firing a driver).
   f. The message itself states that the standard self-service path is already blocked or already failed -- e.g. can't log in to file the report, no driver contact info shown, locked out of the account needed to resolve this same issue (a circular dependency), a prior support contact was already unhelpful/evasive.
   g. A large disputed amount, or a security/account-takeover concern.
   h. The retrieved historical evidence is weak, generic-but-irrelevant, or doesn't actually match this specific message's situation.
   If NONE of (a)-(h) apply, AUTO-HANDLE, even for a real, valid complaint -- that is the expected, correct outcome for most routine disputes.

   In `escalation_reason`, ALWAYS give a real justification: if escalating, name which trigger (a)-(h) applied; if auto-handling, state that no trigger applied. Never leave this as "None" or empty.

3. Draft a short, brand-appropriate reply.
   - Ground it in the RETRIEVED HISTORICAL EVIDENCE you're given -- real past Uber replies to similar messages. Most real Uber replies are short redirects to a help link or a request to DM account details; matching that pattern when the evidence supports it is CORRECT, not lazy.
   - NEVER invent facts, policies, order/trip/payment details, or promises. NEVER claim you completed an action, looked up an account, or issued a refund. NEVER give a specific timeline the evidence doesn't support.
   - If escalating, the reply should acknowledge the issue and say a specialist will follow up -- do not attempt to resolve it yourself.
   - If the evidence is generic, your reply should be too -- do not fabricate specificity the real historical data doesn't have.

Respond with the structured fields only."""


def build_user_prompt(message, evidence):
    if evidence:
        evidence_text = "\n\n".join(
            f"[{i + 1}] similarity={e['similarity']}\n"
            f"    Past customer message: {e['historical_customer_message']}\n"
            f"    Uber's real reply: {e['historical_uber_response']}"
            for i, e in enumerate(evidence)
        )
    else:
        evidence_text = "(no relevant historical evidence retrieved)"

    return f"""CUSTOMER MESSAGE:
{message}

RETRIEVED HISTORICAL EVIDENCE (top-{len(evidence)} similar past conversations):
{evidence_text}"""


def _call_with_retry(client, contents, config):
    """Free-tier Gemini quotas are rate-limited (RPM); retry on 429/5xx with
    backoff rather than failing the whole 200-example run over a transient limit.
    A 429 on this key means the 5-requests/minute free-tier quota was hit
    (discovered live -- see MODEL comment) -- wait out a full window, not a
    short exponential backoff, since a short retry will just 429 again."""
    delay = 15.0
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                print(f"  Rate limited (free-tier quota), waiting {delay:.0f}s...")
                time.sleep(delay)
                continue
            print(f"  Non-retryable client error {e.code} {e.status}: {e.message}")
            raise
        except genai_errors.ServerError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def run_agent(client, corpus, message):
    evidence = retrieve(corpus, message)
    response = _call_with_retry(
        client,
        contents=build_user_prompt(message, evidence),
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            response_mime_type="application/json",
            response_schema=AgentDecision,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    decision = response.parsed

    # Defensive guard: never rely on the prompt alone for a safety-critical policy.
    if decision.intent == "driver_conduct_and_safety" and not decision.should_escalate:
        decision.should_escalate = True
        decision.escalation_reason = (
            decision.escalation_reason
            + " [code-level override: driver_conduct_and_safety is always escalated]"
        )

    return decision, evidence


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def main():
    print("=" * 70)
    print("STEP 7: AI SUPPORT AGENT")
    print("=" * 70)

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    print(f"Test set: {len(golden_set)} examples")

    print("Building leakage-safe TF-IDF retrieval corpus...")
    corpus = build_retrieval_corpus()
    print(f"Retrieval corpus: {len(corpus['texts']):,} historical (query, reply) pairs")

    cache = load_cache()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    new_calls = 0
    for i, example in enumerate(golden_set, 1):
        golden_id = example["golden_id"]
        if golden_id in cache:
            continue
        decision, evidence = run_agent(client, corpus, example["customer_message"])
        cache[golden_id] = {
            "predicted_intent": decision.intent,
            "intent_confidence": decision.intent_confidence,
            "predicted_should_escalate": decision.should_escalate,
            "escalation_reason": decision.escalation_reason,
            "draft_response": decision.draft_response,
            "retrieved_evidence": evidence,
        }
        new_calls += 1
        save_cache(cache)  # checkpoint after every call -- free-tier quotas make
                            # a mid-run failure/interruption realistic, and this
                            # keeps a crash from losing already-paid-for progress
        if new_calls % 10 == 0:
            print(f"  ...{i}/{len(golden_set)} processed ({new_calls} new API calls)")
        time.sleep(CALL_PACING_SECONDS)  # pacing: stay under the 5 req/min free-tier quota

    save_cache(cache)
    print(f"\n{new_calls} new API call(s) made; {len(golden_set) - new_calls} served from cache.")

    results = [
        {"golden_id": ex["golden_id"], "gold_intent": ex["gold_intent"],
         "gold_should_escalate": ex["should_escalate"], **cache[ex["golden_id"]]}
        for ex in golden_set
    ]
    os.makedirs("data/results", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()

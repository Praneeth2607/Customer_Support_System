"""
Step 9: LLM-as-a-Judge Response Quality Evaluation

Scores every one of the 200 AI agent draft responses (data/results/
ai_agent_results.json) against the versioned rubric (src/evaluation/rubric.py,
v1.0) on six dimensions, 1-5 each: correctness, groundedness, helpfulness,
safety, brand_consistency, tone.

Uses the same Gemini setup as the agent itself (gemini-flash-lite-latest,
free tier, checkpointed cache, paced calls) -- see src/pipeline/support_agent.py
and Decision 13 for why.

The judge sees exactly what the agent saw (customer message + retrieved
historical evidence) plus the agent's own draft response -- it does NOT see
the gold_intent/gold_should_escalate labels, since it is judging response
TEXT quality, not classification correctness.

Output: data/judge/llm_judge_results.json (all 200), plus
src/evaluation/judge_agreement.py separately computes human-agreement
statistics on the 50-example overlap with data/judge/human_ratings.json.
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from src.evaluation.rubric import render_rubric_text, RUBRIC_VERSION

load_dotenv()

RESULTS_PATH = "data/results/ai_agent_results.json"
GOLDEN_SET_PATH = "data/golden/golden_evaluation_set.json"
CACHE_PATH = "data/judge/llm_judge_cache.json"
OUTPUT_PATH = "data/judge/llm_judge_results.json"
MODEL = "gemini-flash-lite-latest"
MAX_RETRIES = 8
CALL_PACING_SECONDS = 5.0

class JudgeScores(BaseModel):
    correctness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    brand_consistency: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)
    reasoning: str


def build_judge_system_prompt():
    return f"""You are an impartial quality judge for AI-drafted customer support replies at Uber.

You will be given a customer message, the historical evidence the drafting system was given (real past Uber support conversations retrieved for grounding), and the system's drafted reply. Score the reply on each dimension below using the exact rubric. Be a strict, consistent grader -- do not default to high scores. A generic-but-safe reply that mirrors the real historical pattern should score well; a reply that invents unsupported specifics, ignores the customer's actual point, or copies a detail (like a name) from the evidence that doesn't belong to this customer should score poorly on the relevant dimension.

{render_rubric_text()}

In `reasoning`, give one or two sentences citing the SPECIFIC thing that drove your lowest score (or confirm nothing was wrong if all scores are high). Respond with the structured fields only."""


def build_judge_user_prompt(message, evidence, draft_response):
    if evidence:
        evidence_text = "\n\n".join(
            f"[{i + 1}] Past customer message: {e['historical_customer_message']}\n"
            f"    Uber's real reply: {e['historical_uber_response']}"
            for i, e in enumerate(evidence)
        )
    else:
        evidence_text = "(no historical evidence was retrieved)"

    return f"""CUSTOMER MESSAGE:
{message}

HISTORICAL EVIDENCE GIVEN TO THE DRAFTING SYSTEM:
{evidence_text}

SYSTEM'S DRAFTED REPLY:
{draft_response}"""


def _call_with_retry(client, contents, config):
    delay = 15.0
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                print(f"  Rate limited, waiting {delay:.0f}s...")
                time.sleep(delay)
                continue
            print(f"  Non-retryable client error {e.code} {e.status}: {e.message}")
            raise


def judge_one(client, message, evidence, draft_response):
    response = _call_with_retry(
        client,
        contents=build_judge_user_prompt(message, evidence, draft_response),
        config=types.GenerateContentConfig(
            system_instruction=build_judge_system_prompt(),
            response_mime_type="application/json",
            response_schema=JudgeScores,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    return response.parsed


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
    print(f"STEP 9: LLM-AS-A-JUDGE (rubric v{RUBRIC_VERSION})")
    print("=" * 70)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        agent_results = json.load(f)
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_by_id = {g["golden_id"]: g for g in json.load(f)}
    print(f"Judging {len(agent_results)} draft responses")

    cache = load_cache()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    new_calls = 0
    for i, r in enumerate(agent_results, 1):
        gid = r["golden_id"]
        if gid in cache:
            continue
        message = golden_by_id[gid]["customer_message"]
        scores = judge_one(client, message, r["retrieved_evidence"], r["draft_response"])
        cache[gid] = scores.model_dump()
        new_calls += 1
        save_cache(cache)
        if new_calls % 10 == 0:
            print(f"  ...{i}/{len(agent_results)} processed ({new_calls} new API calls)")
        time.sleep(CALL_PACING_SECONDS)

    save_cache(cache)
    print(f"\n{new_calls} new judge call(s); {len(agent_results) - new_calls} served from cache.")

    output = [
        {"golden_id": r["golden_id"], "rubric_version": RUBRIC_VERSION, **cache[r["golden_id"]]}
        for r in agent_results
    ]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    dims = ["correctness", "groundedness", "helpfulness", "safety", "brand_consistency", "tone"]
    print("\nMean judge scores across all 200:")
    for dim in dims:
        avg = sum(o[dim] for o in output) / len(output)
        print(f"  {dim:20s} {avg:.2f}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

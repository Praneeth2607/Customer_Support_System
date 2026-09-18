# Hiver AI Customer Support System

An end-to-end, trustworthy AI customer support agent with empirical evaluation, intent classification, historical case grounding, and safe escalation arbitration.

Built for the **Hiver SDE Intern Take-Home Assignment**.

---

## 📌 Project Overview

* **Core Premise**: Turning messy, real-world customer support data (Customer Support on Twitter) into a reliable AI agent where *"the proof is worth more than the system."*
* **Core Capabilities**:
  1. **Intent Classification**: Classifying incoming customer queries into an empirical, data-discovered intent taxonomy.
  2. **Grounded Response Generation**: Drafting brand responses strictly grounded in historical resolution patterns.
  3. **Escalation & Safety Arbitration**: Determining whether a query can be safely auto-handled or must be escalated to a human agent, providing an explicit reason.
  4. **Rigorous Evaluation Harness**: Comparing against two baselines (Majority Class & TF-IDF + Logistic Regression), an isolated 150–250 case Golden Evaluation Set, and an LLM-as-a-judge system calibrated against human judgement.

**Reproduce the headline results in one command:**

```bash
python evaluate.py
```

Runs in ~1-2 minutes (regenerates the two baselines fresh; scores the already-committed AI agent results and LLM-judge scores against the frozen golden set — no API key needed for this fast path) and writes `data/results/evaluation_report.md` / `.json`. Pass `--regenerate-agent` and/or `--regenerate-judge` to also re-run the LLM agent/judge themselves (needs `GOOGLE_API_KEY` in `.env`; ~15-20 minutes each on a free-tier key — see Environment Setup below).

---

## 🗺️ Project Roadmap & Progress Tracker

| Step | Description | Status | Key Artifacts |
|---|---|---|---|
| **Step 1** | **Dataset Audit & Schema Inspection** | 🟢 **Completed** | `src/data/audit_dataset.py`, `data/audit/audit_summary.json`, `tests/test_audit.py` |
| **Step 2** | **Uber Support Brand Extraction & Audit** | 🟢 **Completed** | `src/data/extract_uber.py`, `data/processed/uber_tweets.csv`, `tests/test_extraction.py` |
| **Step 3** | **Conversation Thread Reconstruction** | 🟢 **Completed** | `src/data/reconstruct_conversations.py`, `data/processed/uber_conversations.json`, `tests/test_reconstruction.py` |
| **Step 4** | **Intent Discovery & Taxonomy Definition** | 🟢 **Completed** | `src/classification/discover_intents.py`, `docs/intent_taxonomy.md`, `tests/test_taxonomy.py` |
| **Step 5** | Golden Evaluation Set (150–250 cases) | 🟢 **Completed** (frozen) | `data/golden/golden_evaluation_set.json`, `src/evaluation/`, `docs/golden_set_methodology.md` |
| **Step 6** | Baseline Models (Majority + TF-IDF + LogReg) | 🟢 **Completed** | `src/classification/baselines.py`, `data/results/` |
| **Step 7** | Retrieval & AI Support Agent Pipeline | 🟢 **Completed** | `src/pipeline/support_agent.py`, `src/evaluation/evaluate_agent.py`, `data/results/` |
| **Step 8** | Automated Evaluation Harness | 🟢 **Completed** | `evaluate.py`, `src/evaluation/eval_harness.py`, `data/results/evaluation_report.md` |
| **Step 9** | LLM-as-a-Judge Calibration with Human Agreement | 🟢 **Completed** | `src/evaluation/rubric.py`, `llm_judge.py`, `judge_agreement.py`, `docs/response_quality_rubric.md` |
| **Step 10** | Failure Mode Analysis (Top 5 Failures & Hypotheses) | 🟢 **Completed** | `reports/failure_analysis.md`, `tests/test_failure_analysis.py` |
| **Step 11** | Final Report (problem framing, results vs. baselines, failure modes, misleading-number discussion) | 🟢 **Completed** | `reports/final_report.md`, `tests/test_final_report.py` |
| **Step 12** | Decision Log & README Finalized, Reproducibility Verified | 🟢 **Completed** | `docs/DECISION_LOG.md`, `README.md` |

---

## 📊 Step 1 Findings: Dataset Audit Summary

The raw dataset (`twcs.csv`) was fully scanned in memory-efficient 200,000-row chunks:

* **File Size**: 492.58 MB (516,508,641 bytes)
* **Total Rows**: 2,811,774 tweets
* **Inbound vs. Outbound**:
  * **Inbound (Customers)**: 1,537,843 tweets (54.69%)
  * **Outbound (Brands)**: 1,273,931 tweets (45.31%)
* **Schema & Missing Values**:
  * `tweet_id` (int): 0 nulls (100% complete)
  * `author_id` (str): 0 nulls (anonymized user ID for customers; brand handle for brands)
  * `inbound` (bool): 0 nulls
  * `created_at` (str): 0 nulls (RFC 2822 timestamps)
  * `text` (str): 0 nulls (raw tweet text)
  * `response_tweet_id` (str): 1,040,629 nulls (37.01%) — comma-separated IDs of tweets replying to this tweet
  * `in_response_to_tweet_id` (float/int): 794,335 nulls (28.25%) — indicates **Root / Initiating Tweets** (start of conversation)
* **Brand Representation**:
  * **Rank 1**: AmazonHelp (169,840 tweets)
  * **Rank 2**: AppleSupport (106,860 tweets)
  * **Rank 3**: **Uber_Support** (56,270 outbound tweets)
  * Distinct brand handle for Uber: `Uber_Support`

---

## 🚕 Step 2 Findings: Uber Extraction & Conversation Graph Summary

Extracted using a two-pass graph-aware collection algorithm:
* **Total Extracted Tweets**: **125,528** rows (`data/processed/uber_tweets.csv`, 21.86 MB)
  * **Customer Inbound Tweets**: 69,258 (55.17%)
  * **Uber Outbound Responses**: 56,270 (44.83%)
* **Unique Customers**: **39,459** distinct users
* **Conversation Graph Linkages**:
  * **Total Threads**: 42,661 conversation trees
  * **Usable Paired Conversations**: **42,607** (99.87% of all threads have at least 1 customer tweet and 1 Uber reply)
  * **Multi-Turn Conversations ($\ge 3$ tweets)**: 14,284 threads (33.48%)
  * **Unanswered Customer Inquiries**: Only 54 threads
* **Thread Length Distribution**:
  * **Mean**: 2.94 tweets per thread
  * **Median**: 2.0 tweets per thread
  * **Max**: 385 tweets in a single high-engagement thread
* **Text Repetition & Quality**:
  * **Customer Text Duplicate Rate**: 2.58% (exceptionally high lexical diversity)
  * **Uber Outbound Duplicate Rate**: 0.13% (minimal copy-paste bot spam)

---

## 💬 Step 3 Findings: Conversation Reconstruction & Normalization Summary

Reconstructed and normalized into chronological, structured dialogue objects:
* **Total Usable Conversations**: **42,368** complete dialogues (`data/processed/uber_conversations.json`, 81.83 MB)
* **Quality Filtering**:
  * Filtered out 71 threads with no brand reply.
  * Filtered out 145 low-signal tweets consisting purely of `@Uber_Support` mentions without any problem description.
* **Conversational Depth**:
  * **2-Turn Direct Interactions** (Customer $\rightarrow$ Uber): **29,042** conversations (68.55%)
  * **Multi-Turn Dialogues** ($\ge 3$ turns): **13,326** conversations (31.45%)
  * **Average Turns per Dialogue**: **2.60**
* **Linguistic & Content Richness**:
  * **Average Initial Customer Query Length**: **21.37 words** (substantive customer context)
  * **DM Deflection Rate**: **37.62%**
  * **Help URL Grounding Rate**: **63.06%** (brand embeds actionable support/resolution links)
* **Text Normalization Applied**:
  * Stripped noisy leading Twitter handles (`@Uber_Support`, `@115872`) while preserving text semantic tokens.
  * Unescaped HTML entities (`&amp;` $\rightarrow$ `&`).
  * Replaced volatile `t.co` shortened links with canonical `[URL]` tokens.

---

## 🎯 Step 4 Findings: Intent Taxonomy Summary

Informed by TF-IDF n-gram extraction + K-Means clustering (k=10) across the full pool of **42,186** substantive customer inquiries (see [`data/audit/intent_discovery_report.json`](file:///c:/Users/study/OneDrive/Desktop/Projects/Customer_Support_System/data/audit/intent_discovery_report.json)), then finalized by qualitative, domain-informed labelling — see the "Honesty note on provenance" in `docs/intent_taxonomy.md` for exactly what the clustering did and didn't show:

| Intent Name | Core Problem Space | Sample Indicator Keywords | Default Escalation Policy |
|---|---|---|---|
| **`cancellation_issue`** | Cancellation fees, driver no-show cancels | `cancel`, `cancellation fee`, `charged 5` | Conditional Auto-Handle (provide waiver flow) |
| **`fare_and_payment_dispute`** | Overcharges, double billing, tolls, surge | `charged twice`, `overcharged`, `refund`, `fare` | Conditional Auto-Handle (escalate if >$50 or fraud) |
| **`lost_item`** | Phone, wallet, keys left in vehicle | `left phone`, `lost wallet`, `back seat` | **Auto-Handle** (direct driver contact portal) |
| **`driver_conduct_and_safety`** | Reckless driving, verbal abuse, harassment | `unsafe`, `reckless`, `rude`, `threatened` | **MANDATORY ESCALATE (100%)** (Route to safety team) |
| **`pickup_and_route_issue`** | Wrong pickup, detours, car not moving | `wrong route`, `detour`, `refused to go` | Conditional Auto-Handle (route review link) |
| **`account_and_promo_issue`** | Promo code failure, login/2FA, Ride Pass | `promo code`, `discount`, `login locked` | Conditional Auto-Handle (escalate if deactivated) |
| **`out_of_scope_or_unclear`** | Uber Eats / food delivery, or no stated problem | `uber eats`, `order food`, `need help` | **MANDATORY ESCALATE** (redirect or clarify) |

* **Multi-Intent Precedence**: `Safety` > `Lost Item` > `Cancellation` > `Fare Dispute` > `Route` > `Account/Promo` > `Out of Scope/Unclear`.
* **Why a 7th intent**: the k=10 clustering on the full query pool directly surfaces Uber Eats content (~10.7% of traffic) and generic, low-signal messages (~9.4%) as distinct clusters — neither fits a ride-support policy, so forcing them into one of the six ride intents would mislabel ~20% of real inbound traffic. See Decision 9 in `docs/DECISION_LOG.md`.
* Complete taxonomy documented in [`docs/intent_taxonomy.md`](file:///c:/Users/study/OneDrive/Desktop/Projects/Customer_Support_System/docs/intent_taxonomy.md) and coded in [`src/classification/taxonomy.py`](file:///c:/Users/study/OneDrive/Desktop/Projects/Customer_Support_System/src/classification/taxonomy.py).

---

## 🧪 Step 5 Findings: Golden Evaluation Set (Frozen, Hand-Labelled)

* **200 examples** stratified-sampled from a 41,784-row frame (42,186 usable customer queries, minus 402 near-duplicates) using a floor(18) + proportional-top-up quota per intent, then **hand-labelled** in full (`gold_intent`, `should_escalate`, `escalation_reason`, `labeling_notes` for every example) and frozen as `data/golden/golden_evaluation_set.json`.
* **Gold intent distribution**: `out_of_scope_or_unclear` 58, `account_and_promo_issue` 36, `cancellation_issue` 27, `fare_and_payment_dispute` 26, `driver_conduct_and_safety` 21, `lost_item` 16, `pickup_and_route_issue` 16.
* **Escalation**: 130/200 (65%) `should_escalate=true`. `driver_conduct_and_safety` escalates 21/21 (100%, mandatory policy, no exceptions); `out_of_scope_or_unclear` escalates 55/58. **This rate reflects the deliberately rare-intent-heavy sampling design, not real-world escalation volume** — see methodology doc.
* **Heuristic vs. hand-label agreement: 148/200 (74.0%)** — the sampling heuristic's 26% error rate was overwhelmingly Eats content that coincidentally matched a ride-intent keyword (e.g. "cancelled my order"), corrected to `out_of_scope_or_unclear` on manual review. This is itself useful evidence that a naive keyword classifier is a weak baseline.
* One genuine **taxonomy precedence gap** was found and documented rather than silently patched (`golden_0091`: hacked account causing fraudulent charges) — flagged for the Step 10 failure analysis.
* **Leakage control**: `data/golden/golden_conversation_ids.json` reserves all 200 `conversation_id`s for exclusion from any Step 7 retrieval corpus.
* Full methodology, quota derivation, and hand-labelling rules documented in [`docs/golden_set_methodology.md`](file:///c:/Users/study/OneDrive/Desktop/Projects/Customer_Support_System/docs/golden_set_methodology.md); every non-obvious labelling decision also logged in `docs/DECISION_LOG.md` (Decisions 10-11).

---

## 📈 Step 6 Findings: Baseline Intent Classifiers

Both baselines evaluated on the exact same frozen `data/golden/golden_evaluation_set.json` (200 examples), trained on the Step 5 heuristic pseudo-labels applied to ~42k conversations (golden set `conversation_id`s **and** duplicate-text matches excluded — see Decision 12 for a leakage bug this caught and fixed):

| Baseline | Accuracy | Macro F1 |
|---|---|---|
| Majority class (always `out_of_scope_or_unclear`) | 29.0% | 0.0642 |
| TF-IDF + Logistic Regression | 71.0% | 0.7238 |

* TF-IDF+LogReg's weakest class is `out_of_scope_or_unclear` itself (F1 0.5631) — consistent with that intent carrying the heuristic training labels' own worst noise (74% heuristic/gold agreement measured in Step 5).
* **Important caveat**: both baselines are trained on *heuristic* weak labels, not gold labels (no larger hand-labelled corpus exists) — their accuracy partly reflects how well 42k weak labels approximate reality, not just classifier quality. Full discussion in `docs/DECISION_LOG.md` (Decision 12).
* Full per-intent precision/recall/F1 and confusion matrices in `data/results/baseline_majority.json` and `data/results/baseline_tfidf_logreg.json`.

---

## 🤖 Step 7 Findings: AI Support Agent

Pipeline: **TF-IDF retrieval** over the leakage-safe ~42k-conversation corpus (top-3 similar historical `(customer message, real Uber reply)` pairs) → **one Gemini call** (`gemini-flash-lite-latest`) that classifies intent, decides AUTO-HANDLE/ESCALATE with a reason, and drafts a grounded reply — evaluated on the exact same frozen `golden_evaluation_set.json` as the baselines:

| System | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 29.0% | 0.0642 |
| TF-IDF + Logistic Regression | 71.0% | 0.7238 |
| **AI Agent (Gemini)** | **87.0%** | **0.8628** |

* **Escalation decision (after a diagnosed prompt fix — see Decision 14)**: **85.0% accuracy** (up from an initial 76.0%), F1 0.877 on `escalate=True`. The first version's system prompt gave escalation criteria as prose, which caused two distinct failure modes: over-escalating routine disputes (treating "needs an account lookup" as grounds to escalate, when giving the standard self-serve redirect *is* the correct auto-handle outcome) and under-escalating cases with real triggers (repeated-pattern language, unauthorized-charge claims, explicit human requests) that were implied but not made explicit. Replacing the prose with an explicit lettered checklist of valid escalation triggers cut false positives 22→7 (-68%) and false negatives 26→23 (-12%) — both directions improved together, not a threshold trade-off. Full diagnosis, fix, and honest before/after in `docs/DECISION_LOG.md` (Decision 14); the pre-fix run is preserved as `data/results/ai_agent_results_v1_baseline.json` for the record.
* `driver_conduct_and_safety` escalated 21/21 (100%) via the model's own reasoning — the code-level mandatory-escalate override never had to fire.
* **Provider note**: built on Google Gemini (free tier), not the Anthropic API, per the project owner's explicit cost decision. Hit two real, live-discovered constraints — a deprecated model (404) and a 20-request/day cap on the first model tried — both documented with the actual fix in `docs/DECISION_LOG.md` (Decision 13), including a retry-handler bug a live run caught before it could silently corrupt results.
* Leakage: the retrieval corpus uses the same conversation-id + duplicate-text exclusion as Step 6 (`src/classification/leakage_utils.py`), re-verified with its own regression test.
* Full per-intent metrics, confusion matrix, and the three-way comparison in `data/results/ai_agent_metrics.json` and `data/results/system_comparison.json`.

---

## 🧮 Step 8 Findings: Automated Evaluation Harness

`evaluate.py` → `src/evaluation/eval_harness.py` consolidates Steps 5–7 into one reproducible report:

* **Fast path** (`python evaluate.py`, no flags): regenerates both Step 6 baselines fresh (~1 min, deterministic, no API calls) and scores the already-committed Step 7 agent results against the frozen golden set. Runs in **~5-15 seconds to a couple of minutes**, well inside the 15-minute target, and needs no `GOOGLE_API_KEY`.
* **`--regenerate-agent`**: opt-in flag to also re-run the LLM agent from scratch (needs the API key, takes real time/cost) — deliberately not the default, so the fast/free reproduction path is never silently blocked on external API availability.
* Output: `data/results/evaluation_report.md` (human-readable) and `.json` (machine-readable) — the three-way intent comparison, the agent's escalation metrics, and explicit notes flagging the two biggest "don't over-read this number" caveats (baseline training-label noise from Decision 12; escalation accuracy reflecting the Decision 14 prompt fix).
* Updated after Step 9 landed: now also aggregates the LLM-judge response-quality scores and human-agreement statistics (`--regenerate-judge` to re-run the judge itself; same committed-by-default pattern as the agent).

---

## ⚖️ Step 9 Findings: LLM-as-a-Judge + Human Agreement

Six-dimension rubric (correctness, groundedness, helpfulness, safety, brand_consistency, tone; 1–5 scale, versioned in `src/evaluation/rubric.py` v1.0) judges all 200 agent responses; a 50-example subset (proportionally stratified from the golden set's own intent distribution) was rated independently by a human **before** seeing any judge scores, to avoid anchoring.

**Mean LLM-judge scores across all 200**: correctness 4.54, groundedness 4.66, helpfulness 4.53, safety 4.92, brand_consistency 4.80, tone 4.76 — the system's responses skew high, largely because most real historical Uber replies are short, safe, generic redirects, and mirroring that pattern (rather than inventing specificity) is the *correct* behavior per the rubric.

**Human vs. LLM-judge agreement** (n=50, 300 paired scores):

| Statistic | Pooled |
|---|---|
| Mean Absolute Disagreement (primary) | 0.67 |
| Spearman r | 0.436 (p<0.001) |
| Weighted (quadratic) Cohen's kappa | 0.385 |
| Exact match rate | 46% |
| Within 1 point | 87.3% |

* **MAD (not Spearman/kappa) is the reported primary statistic**, chosen only after inspecting the data: `safety` and `brand_consistency` have very low score variance (both rater and judge cluster near 5), which makes their correlation-based stats unstable — `safety`'s Spearman r is literally **-0.082**, which read alone would misleadingly suggest disagreement, when `safety` actually has the **lowest raw disagreement of any dimension** (MAD 0.22). Full reasoning in `docs/DECISION_LOG.md` (Decision 16).
* `groundedness` has the strongest agreement (kappa 0.483) — the most objectively checkable dimension. `tone` — the most subjective — has the weakest (kappa 0.209).
* A live smoke test caught the judge independently flagging the same fabrication a human review found (`golden_0100`'s draft copied a customer name, "Annabel," from retrieved evidence into a reply for a different customer) — evidence the judge discriminates rather than rubber-stamping.
* **Honesty note**: the "human" rater is the developer who built the system (no separate annotator pool available), documented plainly in `docs/response_quality_rubric.md` as a real limitation — this is single-rater self-calibration, not inter-rater reliability across a diverse pool.

---

## 🔎 Step 10 Findings: Failure Mode Analysis

Full write-up in [`reports/failure_analysis.md`](file:///c:/Users/study/OneDrive/Desktop/Projects/Customer_Support_System/reports/failure_analysis.md) — five real, data-grounded failure modes, each with the actual example, expected vs. actual behavior, root cause, hypothesis, and a potential fix. Selected for diversity across the pipeline (data quality → intent classification → escalation logic → response generation), not just the numerically largest error buckets:

1. **Escalated-response phrasing ungrounded in evidence** — a direct, traceable side effect of the Decision 14 fix: 7.0% of escalated responses (8/114) have LLM-judge groundedness ≤2 vs. **0%** of auto-handled ones (0/86). Fixing escalation accuracy gave the model a second, ungrounded source of phrasing.
2. **Cross-customer detail leakage** — `golden_0100`'s draft copied a customer's name ("Annabel") from retrieved evidence into a reply for a different customer. Rare (1/200) but the single lowest-groundedness score in the set and a real trust risk.
3. **"Redirect reflex"** — generic redirects substituted for directly-answerable questions (`golden_0075`, `golden_0116`, `golden_0027`), capped by the fact that real historical Uber replies are themselves often non-answers.
4. **Intent misclassification toward `driver_conduct_and_safety` overstates response severity — invisible to both headline metrics.** `golden_0049` and `golden_0159` both have wrong predicted intent and an inappropriately severe response, yet `should_escalate` still matches gold in both (coincidentally). Neither 87.0% intent accuracy nor 85.0% escalation accuracy flags either example — only the Step 9 judge caught it. This is the most important finding in the analysis: proof that no single metric in this project would have surfaced it alone.
5. **Weak heuristic training-label ceiling on the Step 6 baseline** — the same Eats-vs-ride ambiguity (Decision 12, 74.0% heuristic/gold agreement) recurs from training data through live inference, making it the single most recurring root cause in the whole project.
* Every specific number cited in the report is regression-tested against the underlying data (`tests/test_failure_analysis.py`), not just written down.

---

## 📄 Step 11 Findings: Final Report

Full report in [`reports/final_report.md`](file:///c:/Users/study/OneDrive/Desktop/Projects/Customer_Support_System/reports/final_report.md) — problem framing, what "good" means for Uber support, what we chose not to build, results vs. both baselines, the five failure modes, and an honest "what's misleading about my headline number" section.

**The single most important finding in the whole project, surfaced there**: 87.0% intent accuracy and 85.0% escalation accuracy both look fine in aggregate, but one true `driver_conduct_and_safety` case — a message that explicitly says *"Felt unsafe to give [my destination] over phone"* (`golden_0136`) — was misclassified as a routine `cancellation_issue` and **not escalated at all**. A 1-in-200 rate is invisible inside either headline number, but it's exactly the failure a real deployment can't afford. Locked in as a regression test (`tests/test_final_report.py`) alongside the report's other cited numbers, so it can't silently go stale.

---

## 🛠️ Full Step-by-Step Reproduction

For the fast, one-command headline-results reproduction, see `python evaluate.py` at the top of this README. The commands below reproduce every individual step from raw data through failure analysis, in order — useful for auditing any one step in isolation or re-deriving an artifact from scratch.

```bash
# Step 1: Run dataset audit and tests
python src/data/audit_dataset.py
python -m pytest tests/test_audit.py

# Step 2: Extract Uber corpus and run graph audit
python src/data/extract_uber.py
python -m pytest tests/test_extraction.py

# Step 3: Reconstruct conversation dialogues and run validation tests
python src/data/reconstruct_conversations.py
python -m pytest tests/test_reconstruction.py

# Step 4: Run intent discovery and validate taxonomy
python src/classification/discover_intents.py
python -m pytest tests/test_taxonomy.py

# Step 5: Sample golden set candidates, apply hand-labels, and validate
python src/evaluation/sample_golden_set.py
python -m src.evaluation.apply_hand_labels
python -m pytest tests/test_golden_sampling.py tests/test_golden_labels.py

# Step 6: Run baselines (majority-class + TF-IDF/LogReg) and validate
python -m src.classification.baselines
python -m pytest tests/test_baselines.py

# Step 7: Run the AI agent (needs GOOGLE_API_KEY in .env) and score it
# NOTE: results are already committed under data/results/ -- this step's
# LLM generation is a one-time, slow (free-tier-rate-limited), cached pass;
# re-running only re-generates examples missing from the cache (0 if the
# committed cache is present, so a fresh checkout re-runs in seconds).
python -m src.pipeline.support_agent
python -m src.evaluation.evaluate_agent
python -m pytest tests/test_agent_evaluation.py

# Step 8: Reproduce the consolidated headline results in one command
python evaluate.py
python -m pytest tests/test_eval_harness.py

# Step 9: LLM-as-a-judge response quality + human-agreement calibration
python -m src.evaluation.sample_judge_subset   # already committed; re-run only to change n/seed
python -m src.evaluation.llm_judge             # needs GOOGLE_API_KEY; ~15-20 min on free tier
python -m src.evaluation.judge_agreement       # fast, no API calls
python -m pytest tests/test_llm_judge.py

# Step 10: Failure analysis (report only, no code to run -- validated by tests)
python -m pytest tests/test_failure_analysis.py

# Step 11: Final report (report only, no code to run -- validated by tests)
python -m pytest tests/test_final_report.py
```

---

## 📂 Repository Structure

```text
Customer_Support_System/
├── README.md                  # Master documentation and progress tracking (this file)
├── evaluate.py                # Single-command reproducibility entry point (Step 8)
├── requirements.txt
├── sample.csv                 # 100-row sample of the Twitter customer support dataset
├── twcs.csv                   # Full Twitter customer support dataset (~2.81M rows, ~492.6MB)
├── .env                        # GOOGLE_API_KEY (gitignored, not committed -- create your own)
│
├── data/
│   ├── audit/                 # Step 1/4: dataset + intent-discovery audit outputs
│   ├── processed/              # Step 2/3: extracted Uber tweets, reconstructed conversations
│   ├── golden/                 # Step 5: golden evaluation set, sampling artifacts, leakage manifest
│   ├── results/                 # Step 6/7/8: baseline + AI agent results, metrics, evaluation_report.*
│   └── judge/                  # Step 9: LLM-judge scores, human ratings, agreement report
│
├── docs/
│   ├── DECISION_LOG.md          # All 17 non-obvious engineering decisions, in order
│   ├── intent_taxonomy.md       # Step 4: full 7-intent taxonomy + boundary rules
│   ├── golden_set_methodology.md # Step 5: sampling + hand-labelling methodology
│   └── response_quality_rubric.md # Step 9: versioned judge rubric
│
├── reports/
│   └── failure_analysis.md      # Step 10: top 5 failure modes
│
├── src/
│   ├── data/                    # Step 1-3: audit, extraction, conversation reconstruction
│   ├── classification/          # Step 4/6: taxonomy, intent discovery, baselines, leakage utils
│   ├── evaluation/               # Step 5/6/8/9: golden set sampling+labelling, harness, judge
│   └── pipeline/
│       └── support_agent.py     # Step 7: the AI support agent itself
│
└── tests/                        # One test file per step, all passing (pytest tests/)
```

---

## ⚙️ Environment Setup

* **Python**: 3.13.7
* **Compute**: Local GPU Acceleration
* **Dependencies**: `pip install -r requirements.txt` (`pandas`, `numpy`, `scikit-learn`, `pytest`, `google-genai`, `python-dotenv`, `pydantic`)
* **API key (Steps 7 & 9 only)**: create a `.env` file in the repo root with `GOOGLE_API_KEY=<your Google AI Studio key>`. Not needed to reproduce Steps 1-6, 8, or 10. Not needed for Steps 7/9 either if you're just re-scoring the already-committed `data/results/ai_agent_results.json` / `data/judge/llm_judge_results.json` — only needed to regenerate those from scratch (`python -m src.pipeline.support_agent`, `python -m src.evaluation.llm_judge`, or `evaluate.py --regenerate-agent --regenerate-judge`). `.env` is gitignored; never commit it.

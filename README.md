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
| **Step 8** | Automated Evaluation Harness | ⚪ Pending | `evaluation/eval_harness.py` |
| **Step 9** | LLM-as-a-Judge Calibration with Human Agreement | ⚪ Pending | `evaluation/llm_judge.py` |
| **Step 10** | Failure Mode Analysis (Top 5 Failures & Hypotheses) | ⚪ Pending | `reports/failure_analysis.md` |
| **Step 11** | Report, Decision Log & Final Reproducibility Run | ⚪ Pending | `reports/final_report.md`, `docs/DECISION_LOG.md` |

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

## 🧪 Step 5 Findings: Golden Evaluation Set (Frozen, Hand-Labelled)

* **200 examples** stratified-sampled from a 41,784-row frame (42,186 usable customer queries, minus 402 near-duplicates) using a floor(18) + proportional-top-up quota per intent, then **hand-labelled** in full (`gold_intent`, `should_escalate`, `escalation_reason`, `labeling_notes` for every example) and frozen as `data/golden/golden_evaluation_set.json`.
* **Gold intent distribution**: `out_of_scope_or_unclear` 58, `account_and_promo_issue` 36, `cancellation_issue` 27, `fare_and_payment_dispute` 26, `driver_conduct_and_safety` 21, `lost_item` 16, `pickup_and_route_issue` 16.
* **Escalation**: 130/200 (65%) `should_escalate=true`. `driver_conduct_and_safety` escalates 21/21 (100%, mandatory policy, no exceptions); `out_of_scope_or_unclear` escalates 55/58. **This rate reflects the deliberately rare-intent-heavy sampling design, not real-world escalation volume** — see methodology doc.
* **Heuristic vs. hand-label agreement: 148/200 (74.0%)** — the sampling heuristic's 26% error rate was overwhelmingly Eats content that coincidentally matched a ride-intent keyword (e.g. "cancelled my order"), corrected to `out_of_scope_or_unclear` on manual review. This is itself useful evidence that a naive keyword classifier is a weak baseline.
* One genuine **taxonomy precedence gap** was found and documented rather than silently patched (`golden_0091`: hacked account causing fraudulent charges) — flagged for the Step 10 failure analysis.
* **Leakage control**: `data/golden/golden_conversation_ids.json` reserves all 200 `conversation_id`s for exclusion from any Step 7 retrieval corpus.
* Full methodology, quota derivation, and hand-labelling rules documented in [`docs/golden_set_methodology.md`](file:///c:/Users/study/OneDrive/Desktop/Projects/Customer_Support_System/docs/golden_set_methodology.md); every non-obvious labelling decision also logged in `docs/DECISION_LOG.md` (Decisions 10-11).

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

## 🛠️ How to Reproduce Current Step

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
```

---

## 📂 Repository Structure

```text
Customer_Support_System/
├── README.md               # Master documentation and progress tracking
├── sample.csv              # 100-row sample of the Twitter customer support dataset
├── twcs.csv                # Full Twitter customer support dataset (~2.81M rows, ~492.6MB)
├── data/
│   └── audit/              # Audit outputs (JSON summaries, distributions)
├── docs/                   # Engineering documentation and decision log
├── src/
│   └── data/               # Data ingestion, audit, and conversation reconstruction
└── tests/                  # Deterministic tests (verified with pytest)
```

---

## ⚙️ Environment Setup

* **Python**: 3.13.7
* **Compute**: Local GPU Acceleration
* **Dependencies**: `pip install -r requirements.txt` (`pandas`, `numpy`, `scikit-learn`, `pytest`, `google-genai`, `python-dotenv`, `pydantic`)
* **API key (Step 7 only)**: create a `.env` file in the repo root with `GOOGLE_API_KEY=<your Google AI Studio key>`. Not needed to reproduce Steps 1-6, and not needed for Step 7 either if you're just re-scoring the already-committed `data/results/ai_agent_results.json` — only needed to regenerate agent outputs from scratch. `.env` is gitignored; never commit it.

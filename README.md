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
| **Step 5** | Golden Evaluation Set (150–250 cases) | ⚪ *Next Step* | `data/golden/golden_evaluation_set.json` |
| **Step 6** | Baseline Models (Majority + TF-IDF + LogReg) | ⚪ Pending | `src/classification/baselines.py` |
| **Step 7** | Retrieval & AI Support Agent Pipeline | ⚪ Pending | `src/pipeline/support_agent.py` |
| **Step 8** | Automated Evaluation Harness | ⚪ Pending | `evaluation/eval_harness.py` |
| **Step 9** | LLM-as-a-Judge Calibration with Human Agreement | ⚪ Pending | `evaluation/llm_judge.py` |
| **Step 10** | Failure Mode Analysis (Top 5 Failures & Hypotheses) | ⚪ Pending | `reports/failure_analysis.md` |
| **Step 11** | Report, Decision Log & Final Reproducibility Run | ⚪ Pending | `reports/final_report.md`, `docs/DECISION_LOG.md` |

---

## 🎯 Step 4 Findings: Empirical Intent Taxonomy Summary

Derived via TF-IDF n-gram extraction and K-Means clustering across 42,186 customer inquiries:

| Intent Name | Core Problem Space | Sample Indicator Keywords | Default Escalation Policy |
|---|---|---|---|
| **`cancellation_issue`** | Cancellation fees, driver no-show cancels | `cancel`, `cancellation fee`, `charged 5` | Conditional Auto-Handle (provide waiver flow) |
| **`fare_and_payment_dispute`** | Overcharges, double billing, tolls, surge | `charged twice`, `overcharged`, `refund`, `fare` | Conditional Auto-Handle (escalate if >$50 or fraud) |
| **`lost_item`** | Phone, wallet, keys left in vehicle | `left phone`, `lost wallet`, `back seat` | **Auto-Handle** (direct driver contact portal) |
| **`driver_conduct_and_safety`** | Reckless driving, verbal abuse, harassment | `unsafe`, `reckless`, `rude`, `threatened` | **MANDATORY ESCALATE (100%)** (Route to safety team) |
| **`pickup_and_route_issue`** | Wrong pickup, detours, car not moving | `wrong route`, `detour`, `refused to go` | Conditional Auto-Handle (route review link) |
| **`account_and_promo_issue`** | Promo code failure, login/2FA, Ride Pass | `promo code`, `discount`, `login locked` | Conditional Auto-Handle (escalate if deactivated) |

* **Multi-Intent Precedence**: `Safety` > `Lost Item` > `Cancellation` > `Fare Dispute` > `Route` > `Account/Promo`.
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
* **Dependencies**: `pandas`, `numpy`, `pytest`

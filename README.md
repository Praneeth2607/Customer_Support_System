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
| **Step 3** | Conversation Thread Reconstruction | ⚪ *Next Step* | `src/data/reconstruct_conversations.py` |
| **Step 4** | Intent Discovery & Taxonomy Definition | ⚪ Pending | `docs/intent_taxonomy.md` |
| **Step 5** | Golden Evaluation Set (150–250 cases) | ⚪ Pending | `data/golden/golden_evaluation_set.json` |
| **Step 6** | Baseline Models (Majority + TF-IDF + LogReg) | ⚪ Pending | `src/classification/baselines.py` |
| **Step 7** | Retrieval & AI Support Agent Pipeline | ⚪ Pending | `src/pipeline/support_agent.py` |
| **Step 8** | Automated Evaluation Harness | ⚪ Pending | `evaluation/eval_harness.py` |
| **Step 9** | LLM-as-a-Judge Calibration with Human Agreement | ⚪ Pending | `evaluation/llm_judge.py` |
| **Step 10** | Failure Mode Analysis (Top 5 Failures & Hypotheses) | ⚪ Pending | `reports/failure_analysis.md` |
| **Step 11** | Report, Decision Log & Final Reproducibility Run | ⚪ Pending | `reports/final_report.md`, `docs/DECISION_LOG.md` |

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
# 1. Run the dataset audit script
python src/data/audit_dataset.py

# 2. Run deterministic unit tests
python -m pytest tests/test_audit.py
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

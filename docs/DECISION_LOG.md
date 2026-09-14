# Decision Log

This document tracks non-obvious technical, architectural, and data engineering decisions made throughout the project lifecycle.

---

### Decision 1: Chunked Streaming for Dataset Audit & Ingestion
* **Context**: The raw dataset `twcs.csv` is ~492.6 MB and contains 2,811,774 rows. Naively loading the entire CSV into pandas memory creates multi-gigabyte memory spikes and risk of process thrashing, especially on developer machines.
* **Decision**: Implemented chunked generator streaming (`chunksize=200_000`) for all full-dataset scans and brand filtering.
* **Why**: Streaming bounds RAM usage to <300MB, allows progress reporting, and runs in ~36 seconds while preventing out-of-memory errors on arbitrary hardware.
* **Alternatives Considered**: 
  - *Full `pd.read_csv`*: Rejected due to high peak RAM consumption (~2.5 GB) and unsuitability for constrained environments.
  - *SQLite/DuckDB database ingestion*: Considered, but adding external database dependencies for a 1-time extraction phase was deemed premature complexity when chunked pandas satisfies the 36-second requirement with zero extra overhead.

### Decision 2: Brand Handle Verification (`Uber_Support`)
* **Context**: We hypothesized focusing on Uber based on volume, but Twitter datasets often feature multiple sub-handles (e.g., `@Uber_Eats`, `@Uber_Support`, regional handles like `@Uber_UK`).
* **Decision**: Audited outbound tweets (`inbound == False`) across all 2.81M rows. Discovered that the primary, unified brand handle in this corpus is exactly `Uber_Support` (56,270 outbound tweets), making it the 3rd largest brand in the dataset after AmazonHelp and AppleSupport.
* **Why**: Validating the exact handle in code prevents missing customer conversations and confirms Uber has ample sample density while being 3x more computationally tractable than AmazonHelp.

### Decision 3: Selection of `Uber_Support` Over Alternative Brands
* **Context**: The assignment instructs us not to blindly assume Uber is the final choice, but to evaluate whether alternative brands provide higher-quality conversational data for customer-support AI systems.
* **Empirical Comparison Across 2.81M Rows**:

| Brand | Outbound Tweets | Avg Length (chars) | DM Deflection Rate (%) | Help URL Rate (%) | Multi-Turn Follow-up (%) | Core Limitation / Quality Assessment |
|---|---|---|---|---|---|---|
| **AmazonHelp** | 169,840 | 124.0 | 0.64% | 41.30% | 50.21% | **High multilingual noise** (Japanese, Spanish, German, French mixed under one handle); massive product domain. |
| **AppleSupport** | 106,860 | 136.6 | 52.47% | 75.35% | 29.54% | **Severe DM deflection** (>52% boilerplate redirects); repetitive OS/hardware diagnostics. |
| **Uber_Support** | **56,270** | **110.1** | **35.16%** | **51.28%** | **32.05%** | **Goldilocks choice**: Clean English dialogues, distinct ride-hailing taxonomy, balanced auto-handle vs. escalation. |
| **SpotifyCares** | 43,265 | 129.6 | 30.79% | 50.49% | 31.86% | **Low escalation diversity**: Almost exclusively app/cache troubleshooting with minimal safety/financial risk. |
| **Delta** | 42,253 | 103.9 | 16.45% | 15.34% | 28.43% | **Heavy live operational API dependency** (PNR lookups, airport gate changes, real-time flight delays). |

* **Decision**: Formally selected `Uber_Support` as the project's target brand.
* **Why**: 
  1. *Clean Domain Taxonomy*: Ride-hailing cleanly decomposes into non-overlapping intents (fares, cancellations, lost items, driver conduct, pickup issues).
  2. *Realistic Escalation Boundary*: Uber provides a clear distinction between policy-backed auto-handleable queries (lost item procedures, cancellation fee waiver rules) and mandatory human escalations (physical safety incidents, account takeover, fraudulent charges).
  3. *English-First Focus*: Avoids the complex language-filtering required for AmazonHelp.
  4. *Computational Feasibility*: 56k outbound tweets provides thousands of reconstructable conversations while keeping retrieval latency and evaluation runs under 15 minutes.
* **Alternatives Considered**: 
  - *SpotifyCares*: Best runner-up, but rejected because app bugs rarely present meaningful escalation decisions.
  - *AmazonHelp & AppleSupport*: Rejected due to multilingual fragmentation and excessive boilerplate deflection respectively.

### Decision 4: Graph-Aware Two-Pass Extraction for Bidirectional Context
* **Context**: Filtering only rows where `author_id == 'Uber_Support'` yields 56,270 outbound tweets, but completely drops the customer queries that provoked them. Conversely, filtering text for `@Uber_Support` misses follow-up conversation turns where users drop the handle.
* **Decision**: Implemented a two-pass graph-closure extraction:
  1. *Pass 1 (ID Indexing)*: Mapped all Uber tweet IDs, their parent customer tweets (`in_response_to_tweet_id`), their child customer follow-ups (`response_tweet_id`), and direct inbound mentions of `@Uber_Support` (identifying 125,803 candidate IDs).
  2. *Pass 2 (Metadata Extraction)*: Extracted full records for all 125,528 matched rows to `data/processed/uber_tweets.csv`.
* **Why**: Preserves 100% of the conversational graph, resulting in 42,607 complete, bidirectional customer-agent interaction threads while reducing file size from 492.6 MB to 21.86 MB (a 95.6% reduction).
* **Alternatives Considered**: 
  - *Single-pass regex on `@Uber_Support`*: Rejected because subsequent multi-turn replies frequently omit `@Uber_Support`, truncating conversations.
  - *Extracting only 1-turn Q&A pairs*: Rejected because multi-turn follow-ups are needed to understand whether an issue was actually resolved or escalated historically.

### Decision 5: Conversation Unit Representation & Dual-Text Normalization
* **Context**: Customer tweets contain raw Twitter noise: leading handle mentions (`@Uber_Support @105836`), HTML entities (`&amp;`), and volatile shortlinks (`https://t.co/...`). Feeding raw Twitter handles into TF-IDF models or embedding transformers pollutes vector representations with high-frequency arbitrary ID numbers.
* **Decision**: Designed a structured conversation JSON schema that stores both:
  1. `text`: The raw, immutable tweet string (ensuring auditability and exact reference).
  2. `text_clean`: Normalized text with unescaped HTML, stripped leading `@mentions`, and standardized `[URL]` tokens.
  Conversations are structured as chronological sequences from the initial customer inquiry through all intermediate agent replies and follow-ups, capped at 6 turns.
* **Why**: Allows classifiers and retrieval models to focus purely on customer intent semantics (e.g. *"Driver charged me twice for toll"*) while preserving original tweets for full auditing.
* **Alternatives Considered**: 
  - *In-place destructive text cleaning*: Rejected because losing original tweet IDs or URLs prevents verifying ground-truth link citations.
  - *Treating every tweet independently*: Rejected because single tweets lack conversational context (e.g., customer saying *"Yes, that was the one"* makes no sense without the preceding Uber question).

### Decision 6: Pruning Low-Signal / Pure-Mention Inquiries
* **Context**: During conversation reconstruction, we identified 145 threads where the customer's initial tweet consisted purely of `@Uber_Support` or `@115877 @Uber_Support` with zero descriptive text (e.g., tagging the brand in a photo or trying to get attention).
* **Decision**: Formulated an explicit data quality filter discarding conversations where `first_customer_query_clean` is empty.
* **Why**: Tweets with zero semantic content cannot be classified into any intent taxonomy and would degrade evaluation benchmark reliability and classifier training.
* **Alternatives Considered**: 
  - *Creating a "Greeting/Empty" intent*: Rejected because customer support agents cannot take action or auto-handle a message containing only a brand tag with no issue stated.

### Decision 7: Formulating a Compact 6-Intent Taxonomy Over Fine-Grained Classes
* **Context**: Customer support datasets can be split into dozens of hyper-specific micro-intents (e.g., *"lost iPhone"*, *"lost keys"*, *"cleaning fee"*, *"toll fee"*, *"vomit fraud"*). However, overly granular taxonomies suffer from high inter-annotator disagreement, blurred decision boundaries, and poor classifier generalization.
* **Decision**: Synthesized 6 mutually exclusive, operationally actionable intents grounded in empirical TF-IDF/K-Means clustering:
  1. `cancellation_issue`
  2. `fare_and_payment_dispute`
  3. `lost_item`
  4. `driver_conduct_and_safety`
  5. `pickup_and_route_issue`
  6. `account_and_promo_issue`
* **Why**: Every intent in this taxonomy maps directly to distinct business actions and escalation rules, maintaining high classification reliability and clear evaluation rubrics.
* **Alternatives Considered**: 
  - *Merging down to 3 coarse classes (Billing, Ride, Account)*: Rejected because it groups safe auto-handled queries (lost items) with mandatory safety escalations (reckless driving).
  - *15+ granular classes*: Rejected due to high label ambiguity and severe data sparsity on minority classes.

### Decision 8: Deterministic Multi-Intent Precedence Hierarchy
* **Context**: Customer messages often contain overlapping complaints (e.g., *"Driver was rude, cancelled the ride, and charged me $5"*).
* **Decision**: Established a single-label classification framework enforced by a strict priority hierarchy:
  `Safety / Threat` > `Lost Item` > `Cancellation Issue` > `Fare Dispute` > `Route/Pickup` > `Account/Promo`.
* **Why**: Prioritizes passenger safety and urgent personal property recovery above transactional disputes. Avoids the computational complexity of multi-label classification while guaranteeing safety-first arbitration.
* **Alternatives Considered**: 
  - *Multi-label classification*: Considered, but the assignment explicitly specifies classifying into a small set of intents. Introducing multi-label evaluation would complicate metric interpretation without improving escalation safety.





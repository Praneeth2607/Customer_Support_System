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

### Decision 9: Adding a 7th `out_of_scope_or_unclear` Intent and Re-Running Discovery on the Full Query Pool
* **Context**: A handoff audit found that the original Step 4 write-up claimed the 6-intent taxonomy was "derived via TF-IDF n-gram extraction and K-Means clustering," but the actual `intent_discovery_report.json` (k=6, a random 20,000-query subsample of the 42,186 available) told a different story: 47.1% of the clustered sample was noise/gibberish, 11.7% was Uber Eats food-delivery content with no matching intent, and neither `driver_conduct_and_safety` nor `pickup_and_route_issue` appeared as a distinct cluster at all.
* **Investigation**: Re-ran clustering on the full 42,186-query pool at k=8, 10, and 12 to see whether more data and finer granularity would surface cleaner structure. At k=10 on the full pool: fare/cancellation, lost-item, account/security ("disabled"/"hacked"), a pickup-wait cluster ("minutes away", "waiting"), and Uber Eats content all separate out distinctly, while a large diffuse cluster (41.65%) remains — vocabulary-generic messages ("uber", "ride", "driver", "url") that don't cleanly split further under bag-of-words TF-IDF. `driver_conduct_and_safety` and `pickup_and_route_issue` still don't form their own top-level clusters; they're real but comparatively rare, and their signal sits inside the large generic cluster and the driver-focused clusters rather than being separable by frequency alone.
* **Decision**:
  1. Made `discover_intents.py` cluster the full query pool by default (`sample_size=None`) instead of a 20k subsample, and raised `num_clusters` from 6 to 10, since both changes produced measurably cleaner, more interpretable clusters without materially increasing runtime (still well under a minute).
  2. Added a 7th taxonomy label, `out_of_scope_or_unclear`, covering two things that are now directly visible as distinct clusters and were previously being forced into one of the six ride intents or left unaccounted for: (a) Uber Eats / food-delivery content — a different Uber business line this taxonomy doesn't model policy for — and (b) generic, low-signal messages with no stated problem. Its escalation policy is `mandatory_escalate`: a human should redirect Eats traffic or ask a clarifying question, never guess.
  3. Rewrote the provenance language in `docs/intent_taxonomy.md` to state plainly that clustering *informed* the taxonomy rather than mechanically producing it — the six ride-support intents were finalized by qualitative, domain-informed labelling of cluster content, with the two safety/route intents in particular preserved on domain-knowledge grounds despite not dominating any single cluster.
* **Why**: The original framing overstated the clustering evidence and left ~20% of real traffic (Eats + low-signal) with no legitimate label, which would have forced the Step 5 golden set to either mislabel that traffic or silently exclude it — either way understating how much of the real inbound stream a ride-support classifier will actually see and must safely decline to handle.
* **Effect on prior results**: The 6 original ride-support intents, their definitions, and escalation policies are unchanged. `data/processed/uber_conversations.json` and all Step 1-3 numbers are unaffected. Only `data/audit/intent_discovery_report.json` was regenerated (new clusters, still 6 clusters → 10 clusters), and `taxonomy.py` / `intent_taxonomy.md` / `test_taxonomy.py` were updated to add the 7th intent.
* **Alternatives Considered**:
  - *Leave the 6-intent taxonomy as-is and just fix the misleading README wording*: Rejected — it would still leave Eats and low-signal traffic with no legitimate label, which materially affects how representative the Step 5 golden set can be.
  - *Split Eats and low-signal into two separate intents instead of one combined label*: Considered, but both resolve to the same operational action (mandatory escalate, no ride policy applies), and the assignment favors a small, compact taxonomy — splitting them would add a label without changing any downstream behavior.
  - *Force Eats/noise traffic into the nearest existing ride intent*: Rejected — this is exactly the mislabeling the audit flagged; it would corrupt intent-level precision/recall in the eventual evaluation.

### Decision 10: Floor + Proportional-Top-Up Stratified Sampling for the Golden Set (Not Random, Not Purely Proportional)
* **Context**: The golden set (150-250 examples) needs to both reflect real traffic *and* give every intent, especially rare high-stakes ones (`driver_conduct_and_safety`, `pickup_and_route_issue`), enough support to compute a meaningful per-class F1. Pure random sampling would starve those two intents (estimated ~1-2.5% of frame each); pure proportional sampling has the same problem.
* **Decision**: Used a "floor + proportional top-up" quota (18 examples floor per intent, then 74 remaining slots allocated proportional to estimated real frequency) at n=200, further stratified within each intent bucket by difficulty (easy/multi-intent-ambiguous/short-message/long-conversation, ~50/25/15/10 split). Implemented in `src/evaluation/sample_golden_set.py`, seeded (`random_state=42`) for reproducibility.
* **Why**: Guarantees every intent has ≥21 examples (enough for a per-class F1 to be informative) while still weighting the majority of the budget (fare, out-of-scope, cancellation) toward what real traffic actually looks like, and deliberately oversamples the assignment's required difficulty axes (ambiguous, short, long, multi-intent) rather than leaving them to chance.
* **A pseudo-intent heuristic classifier, not the final label, drives stratification**: a keyword-matching heuristic (superset of `taxonomy.py`'s keyword lists) assigns a provisional `pseudo_intent_heuristic` used only to build balanced candidate pools per intent. Spot-checking confirmed this heuristic has real false negatives (a "$0 bill" fare dispute and a "waited 90 mins" pickup complaint were both heuristically bucketed as `out_of_scope_or_unclear` because they didn't contain a fixed keyword phrase). `gold_intent` is left `null` in the output and must be assigned by a careful hand-labelling pass — see `docs/golden_set_methodology.md`.
* **Leakage handling**: (1) near-duplicate collapse in the sampling frame (dropped 402/42,186 near-identical customer messages before sampling) so the golden set isn't padded with template repeats; (2) a leakage manifest (`data/golden/golden_conversation_ids.json`) reserving all 200 `conversation_id`s, to be excluded from any retrieval corpus built in Step 7 — re-verified at that build time, not just asserted here.
* **Effect on results**: New artifacts only (`data/golden/`); no prior Step 1-4 numbers changed.
* **Alternatives Considered**:
  - *Pure random sampling*: Rejected — would make rare-intent metrics statistically meaningless (see Context).
  - *Pure proportional-to-frequency sampling*: Rejected — same failure mode as random, just slightly less severe; still leaves `pickup_and_route_issue` with ~2 examples at n=200.
  - *Equal quota per intent regardless of real frequency*: Considered, but would make the golden set's overall composition unrepresentative of real inbound traffic, which the report needs to be honest about when interpreting headline numbers (Section 23 of the assignment).

### Decision 11: Hand-Labelling Rules — Eats-Always-Out-of-Scope, Split Escalation Policy for Out-of-Scope, and a Documented Taxonomy Gap
* **Context**: With all 200 candidates hand-labelled (`data/golden/golden_evaluation_set.json`, via `src/evaluation/apply_hand_labels.py`), three consistent judgment rules were needed to label 200 real, messy examples reliably rather than case-by-case improvisation.
* **Decision**:
  1. **Any message whose actual subject is Uber Eats/food delivery is labelled `out_of_scope_or_unclear`, regardless of which ride-intent keyword happened to match it.** E.g. "cancelled my order", "charged for missing food" pattern-match `cancellation_issue`/`fare_and_payment_dispute` keywords but are Eats issues. This was the single largest source of correction versus the sampling heuristic: pseudo-intent vs. hand-label agreement was only 74% (148/200), almost entirely from this exact failure mode.
  2. **Within `out_of_scope_or_unclear`, escalation policy is split by sub-type**: the Eats sub-type always escalates (different business line, no policy grounding); the noise sub-type escalates only when there's a real unresolved concern or insufficient information to act safely — a pure feature request, a safely-answerable informational question, or a joke/boast is labelled `should_escalate = false` (safe to auto-acknowledge). This is a refinement of the uniform `mandatory_escalate` policy set in `taxonomy.py` for this intent: the taxonomy's default policy is a reasonable, conservative *default* for an automated system, but a human labeller applying judgment to each example can tell "no ride policy applies" (Eats — always escalate) apart from "nothing needs solving" (idle chatter — safe to close).
  3. **A driver forcing a passenger out of the vehicle is labelled `driver_conduct_and_safety`, not `pickup_and_route_issue`**, and **an extortion-style "won't return my item unless I pay" claim escalates under `driver_conduct_and_safety`**, both per the taxonomy's own documented boundary rules in `docs/intent_taxonomy.md`.
* **A genuine taxonomy gap was found and documented rather than silently patched**: `golden_0091` (account hacked → fraudulent trip charges) is a case where the fixed precedence hierarchy mechanically resolves to `fare_and_payment_dispute` (rank 4) over `account_and_promo_issue` (rank 6), but the root cause is account takeover, not a billing dispute. Labelled `account_and_promo_issue` by explicit human override, with the conflict recorded in `labeling_notes` for discussion in the Step 10 failure analysis — the precedence hierarchy has no rule for "the higher-precedence intent is a *symptom* of the lower-precedence one," and this is exactly one real example of it, not a hypothetical.
* **Why**: These rules make 200 independent judgment calls reproducible and explainable as one policy instead of 200 ad hoc decisions, and the escalation-policy split for `out_of_scope_or_unclear` avoids a real system needlessly routing harmless chatter to a human queue while still being conservative about anything genuinely out of the agent's grounding.
* **Effect on results**: Gold intent distribution differs meaningfully from the pseudo-intent distribution used for sampling — most notably `out_of_scope_or_unclear` grew from 37 (pseudo) to 58 (gold) as Eats-flavored false negatives in other buckets were corrected. `should_escalate` is 130/200 (65%) True, driven heavily by `out_of_scope_or_unclear` (55/58) and the mandatory-escalate `driver_conduct_and_safety` (21/21) — this is a consequence of the floor+top-up sampling design intentionally over-representing rare/high-stakes intents, and **must not be read as an estimate of real-world escalation volume** (flagged for the "misleading headline number" report section).
* **Alternatives Considered**:
  - *Bulk-fill `should_escalate` from the intent's default policy in `taxonomy.py`*: Rejected — this was flagged as unsafe back in the Step 5 methodology doc precisely because message-level risk (amounts, repetition, prior unresolved contact, personnel demands) changes the correct decision within the same intent.
  - *Silently resolve the golden_0091-style precedence conflict by tweaking `INTENT_PRECEDENCE`*: Rejected — changing the taxonomy's precedence rule specifically because of one example is exactly the kind of silent evaluation-design change Section 31 prohibits; documenting the gap and overriding by hand for this one case is the honest path.
  - *Uniform mandatory-escalate for all of `out_of_scope_or_unclear` (no sub-type split)*: Considered for simplicity, but produces an unrealistic system that escalates harmless chit-chat and feature requests, which would itself be a misleading signal about the agent's practical usefulness.

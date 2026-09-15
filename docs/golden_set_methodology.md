# Golden Evaluation Set — Sampling Methodology

This document explains how the 200-example golden evaluation set was
**sampled** from `data/processed/uber_conversations.json`. It covers the
candidate-selection stage only. Hand-labelling (assigning `gold_intent` and
`should_escalate` to each candidate) is a separate step — see
[Hand-Labelling Protocol](#hand-labelling-protocol-not-yet-run) below for
what remains.

Implementation: `src/evaluation/sample_golden_set.py`
Outputs: `data/golden/golden_candidates.json`, `data/golden/golden_conversation_ids.json`,
`data/golden/golden_sampling_summary.json`

---

## Why not random sampling

A uniform random sample of 200 conversations would mirror the real traffic
mix — which is exactly the problem. `driver_conduct_and_safety` and
`pickup_and_route_issue` are the two highest-*stakes* intents (one is a
mandatory-escalate safety category) but among the lowest-*frequency* in raw
volume. A random draw would likely yield single digits of each, making
per-class precision/recall for those intents statistically meaningless —
exactly the opposite of what a safety-relevant intent needs from a
benchmark. The assignment also explicitly calls for deliberate representation
of rare intents, ambiguous/difficult examples, short messages, long
conversations, and multi-intent messages — none of which a random sample
reliably delivers at n=200.

## Target size

**200** (midpoint of the 150–250 range). Chosen because our stratified
design (below) needs at least ~18–20 examples per intent bucket to be
minimally stable for per-class metrics, times 7 intents = ~140, plus enough
headroom to also stratify by difficulty within each bucket without shrinking
any sub-cell to 1–2 examples.

## Step 1 — Sampling frame construction

1. Start from all 42,368 reconstructed conversations.
2. Apply a **heuristic keyword classifier** (`HEURISTIC_KEYWORDS` in
   `sample_golden_set.py` — a superset of `taxonomy.py`'s keyword lists,
   expanded with terms visible in the Step 4 cluster report) to each
   conversation's `first_customer_query_clean`, to get a **pseudo-intent**
   label used only for stratification.
3. **This heuristic is intentionally not the final label and has known low
   recall.** Spot-checking the `out_of_scope_or_unclear` pseudo-bucket found
   real ride-support complaints the keyword list missed — a "$0 bill" fare
   issue, "waited 90 mins" (a pickup/route complaint), a broken gift card
   (account/promo) — all mislabelled as out-of-scope by the heuristic purely
   because they didn't contain one of our fixed keyword phrases. This is
   fine and expected: the heuristic's only job is to make sure the sampling
   frame has *enough* candidates from each rough bucket to draw from; a
   human relabels every candidate by hand afterward (see below). If we had
   used the heuristic's output as the actual `gold_intent`, that would have
   been the exact scientific-integrity failure Section 30/31 warns against —
   we did not.
4. **Near-duplicate collapse**: computed a signature (first 8 alphanumeric
   tokens, lowercased) per customer message and kept only the first
   occurrence of each signature. Dropped 402 near-duplicate messages from
   42,186 candidates → a 41,784-row sampling frame. This stops the golden
   set from being padded with templated/repeated complaints (e.g. many
   near-identical "please help" tweets), which would inflate apparent
   agreement on easy cases without adding real coverage.

Pseudo-intent distribution across the full frame (**for stratification
weights only — not a claim about the true intent distribution**, given the
heuristic's known low recall):

| Pseudo-intent | Count | % of frame |
|---|---|---|
| `out_of_scope_or_unclear` | 26,424 | 63.24% |
| `cancellation_issue` | 5,280 | 12.64% |
| `fare_and_payment_dispute` | 4,977 | 11.91% |
| `account_and_promo_issue` | 2,518 | 6.03% |
| `lost_item` | 1,110 | 2.66% |
| `driver_conduct_and_safety` | 1,030 | 2.47% |
| `pickup_and_route_issue` | 445 | 1.07% |

## Step 2 — Quota design (floor + proportional top-up)

Each of the 7 intents gets a **floor of 18** examples (ensures every class,
however rare, has enough support for a per-class F1 to mean something), then
the remaining 74 slots are allocated **proportional to estimated real
frequency**, weighted toward the higher-volume intents (fare, out-of-scope,
cancellation, account). This keeps the golden set headline numbers
connected to real traffic shape while guaranteeing rare/high-stakes intents
aren't reduced to noise:

| Intent | Floor | Top-up | Total quota |
|---|---|---|---|
| `fare_and_payment_dispute` | 18 | 19 | **37** |
| `out_of_scope_or_unclear` | 18 | 19 | **37** |
| `cancellation_issue` | 18 | 13 | **31** |
| `account_and_promo_issue` | 18 | 10 | **28** |
| `lost_item` | 18 | 5 | **23** |
| `pickup_and_route_issue` | 18 | 5 | **23** |
| `driver_conduct_and_safety` | 18 | 3 | **21** |
| **Total** | | | **200** |

Within `out_of_scope_or_unclear`, the quota is split ~evenly between the
Eats sub-type and the generic/low-signal sub-type.

## Step 3 — Difficulty stratification within each intent bucket

Within each ride-intent bucket, examples are further tagged and sampled
against a target split:

- **easy** (~50%): single clean keyword match, typical length (>6 words, <4 turns)
- **ambiguous** (~25%): matched keywords from 2+ intents (`multi_intent`) — tests
  precedence-hierarchy behavior on real overlapping complaints
- **short_message** (~15%): ≤6 words — tests behavior on minimal-context input
- **long_conversation** (~10%): ≥4 turns — tests grounding/response generation
  on extended, multi-turn context

Sub-quotas degrade gracefully: if a bucket doesn't have enough examples in a
sub-category (e.g. few long multi-turn `driver_conduct_and_safety`
conversations), the shortfall is backfilled from the rest of that intent's
pool rather than left unfilled. The sampling run reported **no quota
shortfalls** — every bucket had enough candidates.

## Step 4 — Actual achieved sample (n=200, seed=42)

| Intent | Achieved |
|---|---|
| `fare_and_payment_dispute` | 37 |
| `out_of_scope_or_unclear` | 37 |
| `cancellation_issue` | 31 |
| `account_and_promo_issue` | 28 |
| `lost_item` | 23 |
| `pickup_and_route_issue` | 23 |
| `driver_conduct_and_safety` | 21 |

| Difficulty tag | Count |
|---|---|
| easy | 87 |
| zero_signal (heuristic found no ride-intent keyword) | 37 |
| multi_intent | 34 |
| short_message | 23 |
| long_conversation | 19 |

Reproducible with `python src/evaluation/sample_golden_set.py` (fixed
`random_state=42`, deterministic given the same input file).

## Leakage prevention

`data/golden/golden_conversation_ids.json` lists the 200 reserved
`conversation_id`s. **This is the primary leakage control**: when the Step 7
AI agent's historical-retrieval corpus is built, it must explicitly exclude
every ID in this manifest, so the agent can never retrieve the exact
historical resolution to a message it is later evaluated on. Near-duplicate
collapse (Step 1) is a secondary control — it also reduces the chance that a
near-identical *variant* of a golden example survives in the retrieval
corpus and leaks the answer indirectly. We will re-verify at Step 7 build
time that no golden `conversation_id` appears in the retrieval index (not
just check it at sampling time), since the retrieval corpus doesn't exist
yet.

## Schema

Each candidate in `golden_candidates.json`:

```
golden_id                          -- e.g. "golden_0001"
conversation_id                    -- links back to uber_conversations.json
customer_message / customer_message_raw
num_turns
conversation_context                -- full message list (speaker, text, text_clean, turn)
historical_resolution               -- last Uber reply in the thread (text_clean)
has_dm_deflection / has_help_url
pseudo_intent_heuristic             -- stratification aid, NOT the gold label
matched_intent_keywords             -- which ride intents' keywords fired
out_of_scope_subtype_heuristic      -- "eats" | "noise" | null
difficulty_category                 -- easy | ambiguous(multi_intent) | short_message | long_conversation | zero_signal
sampling_category                   -- list of tags explaining why this row was picked
gold_intent                         -- NULL, to be hand-labelled
should_escalate                     -- NULL, to be hand-labelled
escalation_reason                   -- NULL, to be hand-labelled
predicted_intent                    -- NULL, filled in later by baselines/agent during evaluation
labeling_notes                      -- NULL, free text for the labeller to record boundary-case reasoning
```

## Hand-Labelling Protocol (complete)

`gold_intent` and `should_escalate` have been hand-labelled for all 200
candidates. Implementation: `src/evaluation/apply_hand_labels.py` merges
`data/golden/_hand_labels.json` (one entry per `golden_id`: `[gold_intent,
should_escalate, escalation_reason, labeling_notes]`) into the candidate
pool and writes the frozen `data/golden/golden_evaluation_set.json`.

Process actually followed:

1. Every one of the 200 `customer_message` values was read in full context
   (`conversation_context`, `historical_resolution`), and assigned exactly
   one `gold_intent` from the 7-intent taxonomy, applying the documented
   precedence hierarchy for genuinely multi-issue messages.
2. `should_escalate` was set per message content, not bulk-copied from the
   intent's default policy — e.g. a routine single cancellation-fee dispute
   is auto-handleable, but the same intent escalates when the message shows
   a repeated pattern ("twice", "three times"), a large/contested amount, a
   prior unresolved support contact, a personnel demand ("fire your
   driver"), or a blocked self-service path (e.g. the standard
   driver-contact flow doesn't work). `driver_conduct_and_safety` is the one
   intent with **zero** exceptions — every one of its 21 examples escalates,
   per the fixed mandatory policy.
3. `labeling_notes` records the reasoning for every boundary-case,
   multi-intent, or reclassification decision, so the *why* is auditable,
   not just the *what*. A few consistent judgment rules emerged and were
   applied uniformly across all 200:
   - **Any message whose actual subject is Uber Eats/food delivery is
     `out_of_scope_or_unclear`**, regardless of which ride-intent keyword
     happened to match it (e.g. "cancelled my *order*", "*charged* for
     missing food" — these are Eats billing/cancellation issues, not ride
     ones). This is the largest source of pseudo-label correction.
   - Within `out_of_scope_or_unclear`, the **Eats sub-type always
     escalates** (different business line, our agent's grounding doesn't
     cover it); the **noise sub-type escalates only if there's a real
     unresolved concern or insufficient information to act safely** — a
     pure feature request, an informational question with a safe factual
     answer, or a joke/boast gets `should_escalate = false` (safe to
     auto-acknowledge, nothing to resolve).
   - A driver forcing a passenger out of the vehicle (at the destination or
     an unintended location) is `driver_conduct_and_safety`, not
     `pickup_and_route_issue` — precedence follows the taxonomy's own
     boundary rule.
   - An extortion-style "won't return my item unless I pay" claim escalates
     under `driver_conduct_and_safety`, per the taxonomy's own documented
     `lost_item` boundary case.
4. One genuine **taxonomy gap** surfaced during labelling and is flagged in
   `labeling_notes` for `golden_0091` (a hacked account causing fraudulent
   trip charges): the fixed precedence hierarchy mechanically resolves this
   to `fare_and_payment_dispute` (rank 4, above `account_and_promo_issue`
   rank 6), but the root cause is account takeover. Labelled
   `account_and_promo_issue` by override, flagged for discussion in the
   Step 10 failure analysis — the precedence hierarchy doesn't have a rule
   for "the higher-precedence intent is a *symptom* of the lower-precedence
   one."
5. The set is now **frozen** per Section 9/31 of the assignment: no further
   edits to labels to chase a better metric on any downstream model.

### Results

Gold intent distribution (**hand-labelled**, differs from the pseudo-intent
distribution used for stratification — see below):

| Intent | Count | Auto-handle | Escalate |
|---|---|---|---|
| `out_of_scope_or_unclear` | 58 | 3 | 55 |
| `account_and_promo_issue` | 36 | 21 | 15 |
| `cancellation_issue` | 27 | 15 | 12 |
| `fare_and_payment_dispute` | 26 | 11 | 15 |
| `driver_conduct_and_safety` | 21 | **0** | **21** |
| `lost_item` | 16 | 9 | 7 |
| `pickup_and_route_issue` | 16 | 11 | 5 |
| **Total** | **200** | **70** | **130** |

**Heuristic pseudo-intent vs. hand-labelled gold_intent agreement: 148/200
(74.0%)**. The 26% disagreement is not labelling noise — it's the expected,
documented failure mode of the sampling heuristic (see "Why not random
sampling" above): mostly Eats content that matched a ride-intent keyword by
coincidence (e.g. "cancelled my order", "left my drink", "charged for
missing food"), reclassified to `out_of_scope_or_unclear` on manual review.
This is exactly why the heuristic was never used as the final label, and
this 74% agreement number is itself useful evidence for the report's
"what's misleading about my headline number" discussion — a
keyword/pseudo-classifier this noisy would be a weak baseline, and 26%
mislabelling shows how easily a naive approach misreads real support
traffic.

`should_escalate` skews escalate-heavy (130/200, 65%) — largely because
`out_of_scope_or_unclear` (58 examples, 55 escalating) and
`driver_conduct_and_safety` (21 examples, 100% escalating) together account
for 79/130 of all escalations. This is a direct, intentional consequence of
the floor+top-up sampling design (Step 2) deliberately over-representing
rare/high-stakes intents — **the golden set's escalation rate should not be
read as an estimate of real-world escalation volume**; see the report's
misleading-headline-number section once Step 10 is written.

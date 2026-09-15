"""
Step 5: Golden Evaluation Set -- Stratified Sampling

Builds the candidate pool for the 150-250 hand-labelled golden set from
data/processed/uber_conversations.json.

This script does NOT assign gold labels. It only selects which conversations
go into the golden set and tags each with a *heuristic* pseudo-intent (for
stratification only) plus a difficulty/sampling category explaining why it
was picked. `gold_intent` and `should_escalate` are left null for a human
(or a careful manual labelling pass) to fill in -- see Decision Log for why
we don't auto-fill them from the heuristic.

Why not pure random sampling (see docs/golden_set_methodology.md for the
full writeup):
  - Real traffic is dominated by a few intents (fare disputes, out-of-scope
    Eats/noise); a random 200-sample would under-represent
    driver_conduct_and_safety and pickup_and_route_issue so badly that
    per-class precision/recall for those intents would be statistically
    meaningless (potentially 2-5 examples).
  - The assignment explicitly asks for rare intents, ambiguous/difficult
    examples, short messages, long conversations, and multi-intent messages
    to be represented on purpose, not left to chance.

Method:
  1. Apply a keyword heuristic (superset of taxonomy.py's keyword lists,
     expanded with terms visible in the Step 4 cluster report) to every
     conversation's first customer message to get a *pseudo*-intent label
     and record every intent whose keywords matched (multi-intent signal).
  2. Deduplicate the sampling frame: exact-duplicate customer messages, and
     near-duplicates (same first-8-token signature), are collapsed to one
     representative -- prevents the golden set from being padded with
     templated/repeated complaints.
  3. Tag each conversation with a difficulty/sampling category:
     zero_signal (no keyword matched any of the 6 ride intents),
     multi_intent (matched 2+ ride intents), short_message (<=6 words),
     long_conversation (>=4 turns), or easy (single clean match, typical
     length).
  4. Draw a stratified sample using fixed quotas per intent bucket (floor +
     proportional top-up -- see QUOTAS below and the methodology doc for the
     numbers), and within each bucket, quotas across the difficulty tags.
  5. Write the candidate pool, plus a leakage manifest of reserved
     conversation_ids that must be excluded from any retrieval corpus built
     in Step 7.
"""

import json
import os
import random
import re
from collections import Counter, defaultdict

INPUT_JSON = "data/processed/uber_conversations.json"
OUTPUT_CANDIDATES = "data/golden/golden_candidates.json"
OUTPUT_LEAKAGE_MANIFEST = "data/golden/golden_conversation_ids.json"
OUTPUT_SUMMARY = "data/golden/golden_sampling_summary.json"
RANDOM_SEED = 42
TARGET_SIZE = 200

# Heuristic keyword lists for STRATIFICATION ONLY (not a classifier, not
# used anywhere downstream except to build a balanced sampling frame).
# Superset of src/classification/taxonomy.py keywords, expanded with terms
# observed directly in data/audit/intent_discovery_report.json clusters.
HEURISTIC_KEYWORDS = {
    "cancellation_issue": [
        "cancel", "cancelled", "cancellation", "no-show", "no show",
    ],
    "fare_and_payment_dispute": [
        "charged", "overcharged", "double charge", "double charged", "fare",
        "receipt", "refund", "toll", "surge", "cleaning fee", "billed", "fee",
    ],
    "lost_item": [
        "left my phone", "lost phone", "left keys", "lost wallet", "left bag",
        "back seat", "forgot", "lost item", "left my", "left in the car",
        "left in uber", "phone in",
    ],
    "driver_conduct_and_safety": [
        "rude", "unsafe", "reckless", "speeding", "yelled", "harassment",
        "scary", "dangerous", "swerving", "drunk", "threatened", "abuse",
        "abused", "assault", "racist", "discriminat", "groped", "harass",
        "creepy", "inappropriate",
    ],
    "pickup_and_route_issue": [
        "wrong pickup", "detour", "wrong route", "refused to go",
        "not moving", "wrong address", "minutes away", "mins away",
        "wrong location", "never showed", "car isn't moving", "stuck",
    ],
    "account_and_promo_issue": [
        "promo", "discount", "login", "log in", "account locked",
        "verification code", "ride pass", "password", "account disabled",
        "hacked", "deactivated", "2fa", "can't sign in",
    ],
}

# Terms that indicate the two out_of_scope_or_unclear sub-cases.
EATS_KEYWORDS = ["uber eats", "ubereats", "order food", "delivery", "deliver",
                  "restaurant", "food", "ordered"]
NOISE_KEYWORDS = ["need help", "please help", "worst customer service",
                    "customer service", "help please", "can you help"]

RIDE_INTENTS = list(HEURISTIC_KEYWORDS.keys())
ALL_INTENTS = RIDE_INTENTS + ["out_of_scope_or_unclear"]

# Floor + proportional top-up quotas (see docs/golden_set_methodology.md).
# Sums to TARGET_SIZE = 200.
QUOTAS = {
    "cancellation_issue": 31,
    "fare_and_payment_dispute": 37,
    "lost_item": 23,
    "driver_conduct_and_safety": 21,
    "pickup_and_route_issue": 23,
    "account_and_promo_issue": 28,
    "out_of_scope_or_unclear": 37,
}
assert sum(QUOTAS.values()) == TARGET_SIZE

# Within-bucket difficulty split targets (approximate; degrades gracefully
# if a bucket doesn't have enough examples in a sub-category).
DIFFICULTY_SPLIT = {
    "easy": 0.50,
    "ambiguous": 0.25,   # zero_signal or multi_intent
    "short_message": 0.15,
    "long_conversation": 0.10,
}


def match_intents(text_lower):
    """Return the set of ride intents whose heuristic keywords appear in text."""
    matched = set()
    for intent, keywords in HEURISTIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.add(intent)
                break
    return matched


def classify_out_of_scope_subtype(text_lower):
    is_eats = any(kw in text_lower for kw in EATS_KEYWORDS)
    is_noise = any(kw in text_lower for kw in NOISE_KEYWORDS)
    if is_eats:
        return "eats"
    if is_noise:
        return "noise"
    return "noise"  # default bucket for genuinely unclear/no-signal text


def pseudo_label(conv):
    text = conv["first_customer_query_clean"]
    text_lower = text.lower()
    matched = match_intents(text_lower)

    if len(matched) == 0:
        # No ride-intent signal at all -> candidate for out_of_scope, unless
        # it's just too short to have any signal (still zero_signal either way).
        subtype = classify_out_of_scope_subtype(text_lower)
        pseudo_intent = "out_of_scope_or_unclear"
        difficulty = "zero_signal"
    elif len(matched) >= 2:
        # Multiple ride intents matched -> resolve with taxonomy precedence
        # for the pseudo-label, but flag as multi_intent for sampling.
        precedence = [
            "driver_conduct_and_safety", "lost_item", "cancellation_issue",
            "fare_and_payment_dispute", "pickup_and_route_issue",
            "account_and_promo_issue",
        ]
        pseudo_intent = next(i for i in precedence if i in matched)
        difficulty = "multi_intent"
        subtype = None
    else:
        pseudo_intent = next(iter(matched))
        subtype = None
        # Also check for explicit Eats mention even if a ride keyword matched
        # (e.g. "Uber Eats driver was rude") -- rare, but keep as ride intent
        # since the complaint concerns a person's conduct/an actual charge.
        word_count = len(text.split())
        if word_count <= 6:
            difficulty = "short_message"
        else:
            difficulty = "easy"

    word_count = len(text.split())
    num_turns = conv["num_turns"]
    if difficulty == "easy" and word_count > 6 and num_turns >= 4:
        difficulty = "long_conversation"

    return pseudo_intent, difficulty, sorted(matched), subtype, word_count


def near_duplicate_signature(text):
    """First 8 alphanumeric tokens, lowercased -- used to collapse near-dupes."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(tokens[:8])


def build_frame(conversations):
    frame = []
    seen_signatures = set()
    exact_dupes = 0
    near_dupes = 0

    for conv in conversations:
        text = conv["first_customer_query_clean"].strip()
        if len(text) <= 5:
            continue  # already filtered in reconstruction, defensive re-check

        sig = near_duplicate_signature(text)
        if sig in seen_signatures:
            near_dupes += 1
            continue
        seen_signatures.add(sig)

        pseudo_intent, difficulty, matched, subtype, word_count = pseudo_label(conv)
        frame.append({
            "conversation": conv,
            "pseudo_intent": pseudo_intent,
            "difficulty": difficulty,
            "matched_intents": matched,
            "out_of_scope_subtype": subtype,
            "word_count": word_count,
        })

    return frame, near_dupes


def stratified_sample(frame, rng):
    by_intent = defaultdict(list)
    for row in frame:
        by_intent[row["pseudo_intent"]].append(row)

    selected = []
    shortfalls = {}

    for intent in ALL_INTENTS:
        pool = by_intent.get(intent, [])
        rng.shuffle(pool)
        quota = QUOTAS[intent]

        if intent == "out_of_scope_or_unclear":
            # Split roughly evenly between eats / noise subtypes.
            eats_pool = [r for r in pool if r["out_of_scope_subtype"] == "eats"]
            noise_pool = [r for r in pool if r["out_of_scope_subtype"] != "eats"]
            eats_quota = quota // 2
            noise_quota = quota - eats_quota
            picked = eats_pool[:eats_quota] + noise_pool[:noise_quota]
            deficit = quota - len(picked)
            if deficit > 0:
                # top up from whichever subtype pool has leftover
                leftover = eats_pool[eats_quota:] + noise_pool[noise_quota:]
                picked += leftover[:deficit]
            if len(picked) < quota:
                shortfalls[intent] = quota - len(picked)
            selected.extend(picked)
            continue

        # Sub-quotas by difficulty tag within this ride-intent bucket.
        by_diff = defaultdict(list)
        for row in pool:
            tag = row["difficulty"]
            if tag == "zero_signal":
                continue  # zero_signal rows are pseudo-labelled out_of_scope, not here
            if tag == "multi_intent":
                by_diff["ambiguous"].append(row)
            else:
                by_diff[tag].append(row)

        sub_quotas = {
            "easy": round(quota * DIFFICULTY_SPLIT["easy"]),
            "ambiguous": round(quota * DIFFICULTY_SPLIT["ambiguous"]),
            "short_message": round(quota * DIFFICULTY_SPLIT["short_message"]),
            "long_conversation": round(quota * DIFFICULTY_SPLIT["long_conversation"]),
        }
        # Fix rounding drift against the bucket's exact quota.
        drift = quota - sum(sub_quotas.values())
        sub_quotas["easy"] += drift

        picked = []
        leftover_pool = []
        for tag, sub_q in sub_quotas.items():
            tag_pool = by_diff.get(tag, [])
            picked.extend(tag_pool[:sub_q])
            leftover_pool.extend(tag_pool[sub_q:])

        if len(picked) < quota:
            # Backfill from any remaining rows in this intent's pool
            # (any difficulty tag) that weren't already picked.
            picked_ids = {id(r) for r in picked}
            backfill = [r for r in pool if id(r) not in picked_ids]
            rng.shuffle(backfill)
            needed = quota - len(picked)
            picked.extend(backfill[:needed])

        if len(picked) < quota:
            shortfalls[intent] = quota - len(picked)

        selected.extend(picked)

    return selected, shortfalls


def build_candidate_record(golden_id, row):
    conv = row["conversation"]
    messages = conv["messages"]
    brand_messages = [m for m in messages if m["speaker"] == "uber"]
    historical_resolution = brand_messages[-1]["text_clean"] if brand_messages else None

    return {
        "golden_id": golden_id,
        "conversation_id": conv["conversation_id"],
        "customer_message": conv["first_customer_query_clean"],
        "customer_message_raw": conv["first_customer_query"],
        "num_turns": conv["num_turns"],
        "conversation_context": messages,
        "historical_resolution": historical_resolution,
        "has_dm_deflection": conv.get("has_dm_deflection"),
        "has_help_url": conv.get("has_help_url"),
        "pseudo_intent_heuristic": row["pseudo_intent"],
        "matched_intent_keywords": row["matched_intents"],
        "out_of_scope_subtype_heuristic": row["out_of_scope_subtype"],
        "difficulty_category": row["difficulty"],
        "sampling_category": build_sampling_tags(row),
        "gold_intent": None,
        "should_escalate": None,
        "escalation_reason": None,
        "predicted_intent": None,
        "labeling_notes": None,
    }


def build_sampling_tags(row):
    tags = [f"pseudo_intent:{row['pseudo_intent']}"]
    if row["difficulty"] in ("multi_intent", "ambiguous"):
        tags.append("multi_intent_candidate")
    if row["difficulty"] == "zero_signal":
        tags.append("zero_signal_candidate")
    if row["difficulty"] == "short_message" or row["word_count"] <= 6:
        tags.append("short_message")
    if row["difficulty"] == "long_conversation" or row["conversation"]["num_turns"] >= 4:
        tags.append("long_conversation")
    if row["pseudo_intent"] in ("driver_conduct_and_safety", "pickup_and_route_issue"):
        tags.append("rare_intent")
    if row["out_of_scope_subtype"]:
        tags.append(f"out_of_scope:{row['out_of_scope_subtype']}")
    return tags


def main():
    rng = random.Random(RANDOM_SEED)

    print("=" * 70)
    print("STEP 5: GOLDEN EVALUATION SET -- STRATIFIED SAMPLING")
    print("=" * 70)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations):,} conversations")

    frame, near_dupes = build_frame(conversations)
    print(f"Sampling frame after near-duplicate collapse: {len(frame):,} "
          f"({near_dupes:,} near-duplicate customer messages dropped)")

    intent_counts = Counter(row["pseudo_intent"] for row in frame)
    print("\nPseudo-intent distribution across the full frame:")
    for intent in ALL_INTENTS:
        n = intent_counts.get(intent, 0)
        pct = round(n / len(frame) * 100, 2)
        print(f"  {intent:30s} {n:6,d}  ({pct}%)")

    selected, shortfalls = stratified_sample(frame, rng)
    rng.shuffle(selected)

    if shortfalls:
        print("\nWARNING -- quota shortfalls (pool too small for requested quota):")
        for intent, deficit in shortfalls.items():
            print(f"  {intent}: short by {deficit}")

    candidates = [build_candidate_record(f"golden_{i+1:04d}", row)
                  for i, row in enumerate(selected)]

    os.makedirs("data/golden", exist_ok=True)
    with open(OUTPUT_CANDIDATES, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    leakage_manifest = {
        "description": (
            "Conversation IDs reserved for the golden evaluation set. "
            "Any retrieval corpus / few-shot example pool built in Step 7 "
            "MUST exclude these conversation_ids to prevent the AI agent "
            "from retrieving the exact answer to a question it is being "
            "evaluated on."
        ),
        "count": len(candidates),
        "conversation_ids": sorted(c["conversation_id"] for c in candidates),
    }
    with open(OUTPUT_LEAKAGE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(leakage_manifest, f, indent=2)

    final_intent_counts = Counter(c["pseudo_intent_heuristic"] for c in candidates)
    final_difficulty_counts = Counter(c["difficulty_category"] for c in candidates)
    turn_dist = Counter(c["num_turns"] for c in candidates)

    summary = {
        "target_size": TARGET_SIZE,
        "actual_size": len(candidates),
        "random_seed": RANDOM_SEED,
        "sampling_frame_size": len(frame),
        "near_duplicates_dropped": near_dupes,
        "quotas_planned": QUOTAS,
        "pseudo_intent_counts_achieved": dict(final_intent_counts),
        "difficulty_counts_achieved": dict(final_difficulty_counts),
        "turn_distribution_achieved": {str(k): v for k, v in sorted(turn_dist.items())},
        "quota_shortfalls": shortfalls,
    }
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSelected {len(candidates)} golden set candidates.")
    print("\nAchieved pseudo-intent distribution (sampled set):")
    for intent in ALL_INTENTS:
        print(f"  {intent:30s} {final_intent_counts.get(intent, 0)}")
    print("\nAchieved difficulty distribution:")
    for tag, n in final_difficulty_counts.items():
        print(f"  {tag:20s} {n}")
    print(f"\nWrote candidates to {OUTPUT_CANDIDATES}")
    print(f"Wrote leakage manifest to {OUTPUT_LEAKAGE_MANIFEST}")
    print(f"Wrote sampling summary to {OUTPUT_SUMMARY}")
    print("\nNOTE: gold_intent / should_escalate are null. Hand-labelling is a")
    print("separate step -- see docs/golden_set_methodology.md.")


if __name__ == "__main__":
    main()

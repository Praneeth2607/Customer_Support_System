"""
Step 9: Sample the human-rating subset for LLM-judge calibration.

Draws n=50 of the 200 AI agent responses (data/results/ai_agent_results.json)
for human rating, stratified proportionally to the golden set's own
gold_intent distribution (already a deliberately-designed stratified sample
from Step 5 -- subsampling it proportionally preserves that design rather
than introducing a new, uncontrolled sampling bias).

n=50 sits inside the assignment's suggested 40-60 range (Section 21).

Output: data/judge/human_rating_subset_ids.json (the 50 golden_ids selected)
"""

import json
import random
from collections import Counter, defaultdict

RESULTS_PATH = "data/results/ai_agent_results.json"
OUTPUT_PATH = "data/judge/human_rating_subset_ids.json"
SUBSET_SIZE = 50
RANDOM_SEED = 42


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    by_intent = defaultdict(list)
    for r in results:
        by_intent[r["gold_intent"]].append(r["golden_id"])

    total = len(results)
    rng = random.Random(RANDOM_SEED)

    # Largest-remainder rounding so per-intent quotas sum to exactly SUBSET_SIZE.
    raw_quotas = {intent: len(ids) / total * SUBSET_SIZE for intent, ids in by_intent.items()}
    floor_quotas = {intent: int(q) for intent, q in raw_quotas.items()}
    remainder = SUBSET_SIZE - sum(floor_quotas.values())
    remainders_sorted = sorted(raw_quotas, key=lambda i: raw_quotas[i] - floor_quotas[i], reverse=True)
    for intent in remainders_sorted[:remainder]:
        floor_quotas[intent] += 1

    selected = []
    for intent, quota in floor_quotas.items():
        pool = by_intent[intent][:]
        rng.shuffle(pool)
        selected.extend(pool[:quota])

    rng.shuffle(selected)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"subset_size": len(selected), "random_seed": RANDOM_SEED,
                   "golden_ids": selected}, f, indent=2)

    print(f"Selected {len(selected)} examples for human rating.")
    print("Per-intent quotas:", dict(floor_quotas))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

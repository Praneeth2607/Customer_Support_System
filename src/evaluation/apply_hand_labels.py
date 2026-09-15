"""
Step 5b: Apply hand-labels to the golden set candidate pool.

Merges data/golden/_hand_labels.json (golden_id -> [gold_intent,
should_escalate, escalation_reason, labeling_notes]) into
data/golden/golden_candidates.json, validates every label against the
taxonomy, and writes the frozen data/golden/golden_evaluation_set.json.

This is a one-time merge, run once the hand-labelling pass is complete.
Per Section 9/31 of the assignment, golden_evaluation_set.json is then
treated as FROZEN -- do not re-run this script after downstream evaluation
has started, and do not hand-edit labels to chase a better metric.
"""

import json
from src.classification.taxonomy import INTENTS

CANDIDATES_PATH = "data/golden/golden_candidates.json"
LABELS_PATH = "data/golden/_hand_labels.json"
OUTPUT_PATH = "data/golden/golden_evaluation_set.json"


def main():
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    candidate_ids = {c["golden_id"] for c in candidates}
    label_ids = set(labels.keys())

    missing_labels = candidate_ids - label_ids
    extra_labels = label_ids - candidate_ids
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} candidates have no hand-label: "
                          f"{sorted(missing_labels)[:5]}...")
    if extra_labels:
        raise ValueError(f"{len(extra_labels)} labels reference unknown golden_ids: "
                          f"{sorted(extra_labels)[:5]}...")

    for c in candidates:
        gold_intent, should_escalate, escalation_reason, notes = labels[c["golden_id"]]
        if gold_intent not in INTENTS:
            raise ValueError(f"{c['golden_id']}: '{gold_intent}' is not a valid intent")
        if not isinstance(should_escalate, bool):
            raise ValueError(f"{c['golden_id']}: should_escalate must be boolean")
        c["gold_intent"] = gold_intent
        c["should_escalate"] = should_escalate
        c["escalation_reason"] = escalation_reason
        c["labeling_notes"] = notes

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    from collections import Counter
    intent_counts = Counter(c["gold_intent"] for c in candidates)
    escalate_counts = Counter(c["should_escalate"] for c in candidates)
    pseudo_vs_gold_agree = sum(
        1 for c in candidates if c["gold_intent"] == c["pseudo_intent_heuristic"]
    )

    print(f"Applied {len(candidates)} hand-labels -> {OUTPUT_PATH}")
    print("\nGold intent distribution:")
    for intent in INTENTS:
        print(f"  {intent:30s} {intent_counts.get(intent, 0)}")
    print(f"\nshould_escalate: True={escalate_counts.get(True, 0)}  "
          f"False={escalate_counts.get(False, 0)}")
    print(f"\nHeuristic pseudo-intent vs. hand-labelled gold_intent agreement: "
          f"{pseudo_vs_gold_agree}/{len(candidates)} "
          f"({round(100 * pseudo_vs_gold_agree / len(candidates), 1)}%)")


if __name__ == "__main__":
    main()

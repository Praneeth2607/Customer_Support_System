"""
Step 9: LLM Judge vs. Human Agreement

Compares the LLM judge's scores (data/judge/llm_judge_results.json) against
the human ratings (data/judge/human_ratings.json) on the 50-example overlap,
per dimension and pooled across all 300 (dimension, example) score pairs.

Reports three agreement statistics rather than picking one in advance --
Section 21 explicitly warns against choosing a statistic because it looks
good, not because it fits the data:
  - Spearman correlation: monotonic-relationship strength, robust to each
    rater's absolute scale, but underpowered when a dimension has little
    score variance (e.g. if everyone scores safety a 5).
  - Weighted (quadratic) Cohen's kappa: chance-corrected ordinal agreement,
    penalizes big misses more than small ones. Needs enough score variety to
    be well-defined -- degenerate (undefined or 0/0) on a near-constant
    dimension.
  - Mean Absolute Disagreement (MAD): |human - judge| averaged. Always
    well-defined regardless of variance, most directly interpretable
    ("scores differ by X points on average"), least statistically
    sophisticated.

Which one is reported as the HEADLINE number is decided after looking at
the actual score distributions (see main()'s printed variance check and
docs/DECISION_LOG.md for the reasoning), not fixed in code ahead of time.
"""

import json
from collections import defaultdict

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

DIMENSIONS = ["correctness", "groundedness", "helpfulness", "safety",
              "brand_consistency", "tone"]

HUMAN_RATINGS_PATH = "data/judge/human_ratings.json"
JUDGE_RESULTS_PATH = "data/judge/llm_judge_results.json"
OUTPUT_PATH = "data/judge/agreement_report.json"


def load_paired_scores():
    with open(HUMAN_RATINGS_PATH, "r", encoding="utf-8") as f:
        human = json.load(f)
    with open(JUDGE_RESULTS_PATH, "r", encoding="utf-8") as f:
        judge_list = json.load(f)
    judge = {j["golden_id"]: j for j in judge_list}

    missing = set(human.keys()) - set(judge.keys())
    if missing:
        raise ValueError(f"{len(missing)} human-rated examples have no judge score: {missing}")

    paired = {dim: {"human": [], "judge": []} for dim in DIMENSIONS}
    for gid, h in human.items():
        j = judge[gid]
        for dim in DIMENSIONS:
            paired[dim]["human"].append(h[dim])
            paired[dim]["judge"].append(j[dim])
    return paired, len(human)


def compute_agreement(paired):
    report = {}
    for dim in DIMENSIONS:
        h, j = paired[dim]["human"], paired[dim]["judge"]
        mad = sum(abs(a - b) for a, b in zip(h, j)) / len(h)

        h_variance = len(set(h)) > 1
        j_variance = len(set(j)) > 1
        if h_variance and j_variance:
            spearman_r, spearman_p = spearmanr(h, j)
        else:
            spearman_r, spearman_p = None, None

        try:
            kappa = cohen_kappa_score(h, j, weights="quadratic")
        except Exception:
            kappa = None

        report[dim] = {
            "n": len(h),
            "mean_absolute_disagreement": round(mad, 4),
            "spearman_r": round(float(spearman_r), 4) if spearman_r is not None else None,
            "spearman_p": round(float(spearman_p), 4) if spearman_p is not None else None,
            "weighted_kappa": round(float(kappa), 4) if kappa is not None else None,
            "human_score_distribution": {str(s): h.count(s) for s in range(1, 6)},
            "judge_score_distribution": {str(s): j.count(s) for s in range(1, 6)},
            "human_unique_values": len(set(h)),
            "judge_unique_values": len(set(j)),
        }
    return report


def pooled_agreement(paired):
    all_h, all_j = [], []
    for dim in DIMENSIONS:
        all_h.extend(paired[dim]["human"])
        all_j.extend(paired[dim]["judge"])
    mad = sum(abs(a - b) for a, b in zip(all_h, all_j)) / len(all_h)
    spearman_r, spearman_p = spearmanr(all_h, all_j)
    try:
        kappa = cohen_kappa_score(all_h, all_j, weights="quadratic")
    except Exception:
        kappa = None
    exact_match_rate = sum(1 for a, b in zip(all_h, all_j) if a == b) / len(all_h)
    within_1_rate = sum(1 for a, b in zip(all_h, all_j) if abs(a - b) <= 1) / len(all_h)
    return {
        "n_score_pairs": len(all_h),
        "mean_absolute_disagreement": round(mad, 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "weighted_kappa": round(float(kappa), 4) if kappa is not None else None,
        "exact_match_rate": round(exact_match_rate, 4),
        "within_1_point_rate": round(within_1_rate, 4),
    }


def main():
    print("=" * 70)
    print("STEP 9: LLM JUDGE vs. HUMAN AGREEMENT")
    print("=" * 70)

    paired, n = load_paired_scores()
    print(f"Comparing on {n} human-rated examples x {len(DIMENSIONS)} dimensions "
          f"= {n * len(DIMENSIONS)} paired scores")

    per_dim = compute_agreement(paired)
    pooled = pooled_agreement(paired)

    print("\nPer-dimension agreement:")
    print(f"{'dimension':20s} {'MAD':>6s} {'Spearman r':>11s} {'Weighted kappa':>15s} {'unique(H/J)':>12s}")
    for dim in DIMENSIONS:
        d = per_dim[dim]
        sr = f"{d['spearman_r']:.3f}" if d['spearman_r'] is not None else "n/a"
        wk = f"{d['weighted_kappa']:.3f}" if d['weighted_kappa'] is not None else "n/a"
        print(f"{dim:20s} {d['mean_absolute_disagreement']:>6.3f} {sr:>11s} {wk:>15s} "
              f"{d['human_unique_values']}/{d['judge_unique_values']:>10}")

    print(f"\nPooled across all {pooled['n_score_pairs']} score pairs:")
    print(f"  Mean Absolute Disagreement: {pooled['mean_absolute_disagreement']}")
    print(f"  Spearman r: {pooled['spearman_r']} (p={pooled['spearman_p']})")
    print(f"  Weighted (quadratic) Cohen's kappa: {pooled['weighted_kappa']}")
    print(f"  Exact match rate: {pooled['exact_match_rate']}")
    print(f"  Within 1 point rate: {pooled['within_1_point_rate']}")

    report = {
        "n_examples": n,
        "dimensions": DIMENSIONS,
        "per_dimension": per_dim,
        "pooled": pooled,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

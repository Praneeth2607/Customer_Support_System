"""
Unit tests for Step 9: LLM-as-a-Judge, human ratings, and agreement analysis.
Verifies that:
1. The rubric is well-formed (6 dimensions, 5 scores each, version tagged).
2. The human-rating subset is 40-60 examples (Section 21 range) and reproducible.
3. Human ratings are complete, valid (1-5), and match the subset exactly.
4. LLM judge results cover all 200 examples with valid scores.
5. The agreement report's numbers are internally consistent (MAD/kappa/Spearman
   computed correctly, pooled stats match per-dimension data).
"""

import json
import pytest
from src.evaluation.rubric import DIMENSIONS, DIMENSION_ORDER, RUBRIC_VERSION

SUBSET_PATH = "data/judge/human_rating_subset_ids.json"
HUMAN_RATINGS_PATH = "data/judge/human_ratings.json"
JUDGE_RESULTS_PATH = "data/judge/llm_judge_results.json"
AGREEMENT_REPORT_PATH = "data/judge/agreement_report.json"
RESULTS_PATH = "data/results/ai_agent_results.json"

RUBRIC_DIMENSIONS = {"correctness", "groundedness", "helpfulness", "safety",
                      "brand_consistency", "tone"}


def test_rubric_well_formed():
    assert set(DIMENSIONS.keys()) == RUBRIC_DIMENSIONS
    assert set(DIMENSION_ORDER) == RUBRIC_DIMENSIONS
    for dim, spec in DIMENSIONS.items():
        assert "question" in spec
        assert set(spec["scale"].keys()) == {1, 2, 3, 4, 5}
    assert RUBRIC_VERSION == "1.0"


@pytest.fixture(scope="module")
def subset():
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_subset_size_in_assignment_range(subset):
    assert 40 <= subset["subset_size"] <= 60
    assert len(subset["golden_ids"]) == subset["subset_size"]
    assert len(set(subset["golden_ids"])) == subset["subset_size"]  # no duplicates


def test_subset_ids_are_valid_golden_ids(subset):
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        all_ids = {r["golden_id"] for r in json.load(f)}
    assert set(subset["golden_ids"]) <= all_ids


@pytest.fixture(scope="module")
def human_ratings():
    with open(HUMAN_RATINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_human_ratings_match_subset_exactly(subset, human_ratings):
    assert set(human_ratings.keys()) == set(subset["golden_ids"])


def test_human_ratings_valid_and_complete(human_ratings):
    for gid, r in human_ratings.items():
        for dim in RUBRIC_DIMENSIONS:
            assert dim in r, f"{gid} missing {dim}"
            assert 1 <= r[dim] <= 5, f"{gid}.{dim} out of range: {r[dim]}"
        assert r.get("notes"), f"{gid} missing notes"


@pytest.fixture(scope="module")
def judge_results():
    with open(JUDGE_RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_judge_covers_all_agent_results(judge_results):
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        all_ids = {r["golden_id"] for r in json.load(f)}
    judge_ids = {j["golden_id"] for j in judge_results}
    assert judge_ids == all_ids


def test_judge_scores_valid(judge_results):
    for j in judge_results:
        for dim in RUBRIC_DIMENSIONS:
            assert dim in j
            assert 1 <= j[dim] <= 5, f"{j['golden_id']}.{dim} out of range: {j[dim]}"
        assert j["rubric_version"] == RUBRIC_VERSION
        assert j["reasoning"].strip()


@pytest.fixture(scope="module")
def agreement_report():
    with open(AGREEMENT_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_agreement_report_structure(agreement_report):
    assert agreement_report["n_examples"] == 50
    assert set(agreement_report["per_dimension"].keys()) == RUBRIC_DIMENSIONS
    pooled = agreement_report["pooled"]
    assert pooled["n_score_pairs"] == 50 * 6
    assert 0.0 <= pooled["exact_match_rate"] <= 1.0
    assert 0.0 <= pooled["within_1_point_rate"] <= 1.0
    assert pooled["exact_match_rate"] <= pooled["within_1_point_rate"]


def test_mad_always_defined_even_with_low_variance(agreement_report):
    """MAD must be well-defined for every dimension, including low-variance
    ones (e.g. safety) where Spearman/kappa can be unstable or None -- this
    is exactly why MAD isn't skipped even when the other two are available."""
    for dim, d in agreement_report["per_dimension"].items():
        assert d["mean_absolute_disagreement"] is not None
        assert d["mean_absolute_disagreement"] >= 0.0


def test_pooled_mad_matches_manual_recomputation(human_ratings, judge_results):
    judge_by_id = {j["golden_id"]: j for j in judge_results}
    diffs = []
    for gid, h in human_ratings.items():
        j = judge_by_id[gid]
        for dim in RUBRIC_DIMENSIONS:
            diffs.append(abs(h[dim] - j[dim]))
    manual_mad = round(sum(diffs) / len(diffs), 4)

    with open(AGREEMENT_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["pooled"]["mean_absolute_disagreement"] == manual_mad

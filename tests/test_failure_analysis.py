"""
Unit tests for Step 10: Failure Mode Analysis.
Verifies that the specific numbers and examples cited in
reports/failure_analysis.md are still accurate against the underlying data --
this report cites real counts (e.g. "8/200 escalated responses have
groundedness<=2"), and if the underlying data changes, that claim must be
caught as stale, not silently left wrong in a written report.
"""

import json


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_failure_mode_1_groundedness_gap():
    """8/114 escalated responses have groundedness<=2; 0/86 auto-handled do."""
    judge = {j["golden_id"]: j for j in load("data/judge/llm_judge_results.json")}
    results = load("data/results/ai_agent_results.json")

    escalated = [r for r in results if r["predicted_should_escalate"]]
    auto = [r for r in results if not r["predicted_should_escalate"]]
    low_g_escalated = sum(1 for r in escalated if judge[r["golden_id"]]["groundedness"] <= 2)
    low_g_auto = sum(1 for r in auto if judge[r["golden_id"]]["groundedness"] <= 2)

    assert len(escalated) == 114
    assert len(auto) == 86
    assert low_g_escalated == 8
    assert low_g_auto == 0


def test_failure_mode_1_cited_examples_are_escalated_and_low_groundedness():
    judge = {j["golden_id"]: j for j in load("data/judge/llm_judge_results.json")}
    results = {r["golden_id"]: r for r in load("data/results/ai_agent_results.json")}
    cited = ["golden_0142", "golden_0063", "golden_0148", "golden_0168",
             "golden_0171", "golden_0177"]
    for gid in cited:
        assert results[gid]["predicted_should_escalate"] is True, gid
        assert judge[gid]["groundedness"] <= 2, gid


def test_failure_mode_2_example_is_lowest_groundedness():
    judge = load("data/judge/llm_judge_results.json")
    lowest = min(judge, key=lambda j: j["groundedness"])
    assert lowest["golden_id"] == "golden_0100"
    assert lowest["groundedness"] == 1


def test_failure_mode_4_cited_examples_have_intent_mismatch_but_correct_escalation():
    results = {r["golden_id"]: r for r in load("data/results/ai_agent_results.json")}
    for gid in ["golden_0049", "golden_0159"]:
        r = results[gid]
        assert r["predicted_intent"] == "driver_conduct_and_safety"
        assert r["gold_intent"] != "driver_conduct_and_safety"
        assert r["predicted_should_escalate"] == r["gold_should_escalate"] == True


def test_failure_mode_5_heuristic_agreement_rate():
    """Cross-check against Decision 12's measured 74.0% (148/200) heuristic/gold
    agreement -- recomputed from the golden set's own pseudo vs gold fields."""
    golden = load("data/golden/golden_evaluation_set.json")
    agree = sum(1 for g in golden if g["pseudo_intent_heuristic"] == g["gold_intent"])
    assert agree == 148
    assert round(100 * agree / len(golden), 1) == 74.0

"""
Unit tests for Step 11: Final Report.
Verifies that the specific numbers cited in reports/final_report.md --
especially the safety-miss finding in Section 6, the most important claim
in the report -- are accurate against the underlying data, so they can't
silently go stale.
"""

import json


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_safety_intent_precision_recall_cited_in_report():
    metrics = load("data/results/ai_agent_metrics.json")
    safety = metrics["intent_classification"]["per_class"]["driver_conduct_and_safety"]
    assert safety["precision"] == 0.75
    assert round(safety["recall"], 3) == 0.857
    assert safety["support"] == 21


def test_true_safety_case_breakdown():
    """18 correctly identified as safety, 2 more misclassified-but-still-escalated
    by coincidence, 1 (golden_0136) misclassified AND not escalated -- the
    single most important finding in the final report."""
    results = load("data/results/ai_agent_results.json")
    true_safety = [r for r in results if r["gold_intent"] == "driver_conduct_and_safety"]
    assert len(true_safety) == 21

    intent_correct = [r for r in true_safety if r["predicted_intent"] == "driver_conduct_and_safety"]
    escalated_anyway = [r for r in true_safety if r["predicted_should_escalate"]]
    missed = [r for r in true_safety if not r["predicted_should_escalate"]]

    assert len(intent_correct) == 18
    assert len(escalated_anyway) == 20
    assert len(missed) == 1
    assert missed[0]["golden_id"] == "golden_0136"
    assert missed[0]["predicted_intent"] == "cancellation_issue"


def test_golden_0136_message_contains_safety_language():
    golden = {g["golden_id"]: g for g in load("data/golden/golden_evaluation_set.json")}
    msg = golden["golden_0136"]["customer_message"].lower()
    assert "unsafe" in msg


def test_golden_set_escalate_rate_cited_as_65_percent():
    golden = load("data/golden/golden_evaluation_set.json")
    escalate_true = sum(1 for g in golden if g["should_escalate"])
    rate = round(100 * escalate_true / len(golden))
    assert rate == 65


def test_groundedness_split_escalated_vs_auto_handled():
    """4.97 auto-handled vs 4.43 escalated, cited in Section 6."""
    judge = {j["golden_id"]: j for j in load("data/judge/llm_judge_results.json")}
    results = load("data/results/ai_agent_results.json")
    escalated = [r for r in results if r["predicted_should_escalate"]]
    auto = [r for r in results if not r["predicted_should_escalate"]]
    avg_g_escalated = sum(judge[r["golden_id"]]["groundedness"] for r in escalated) / len(escalated)
    avg_g_auto = sum(judge[r["golden_id"]]["groundedness"] for r in auto) / len(auto)
    assert round(avg_g_escalated, 2) == 4.43
    assert round(avg_g_auto, 2) == 4.97

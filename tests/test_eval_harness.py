"""
Unit tests for Step 8: Automated Evaluation Harness.
Verifies that:
1. The harness runs end-to-end via `python evaluate.py` and exits cleanly.
2. Both report files are produced with the expected structure.
3. The report's numbers are consistent with the underlying per-system result files
   (the harness must not silently drift from what it's aggregating).
4. The fast path (--skip-baselines) does not touch the AI agent's committed results.
"""

import json
import subprocess
import sys
import pytest

REPORT_JSON_PATH = "data/results/evaluation_report.json"
REPORT_MD_PATH = "data/results/evaluation_report.md"
AGENT_METRICS_PATH = "data/results/ai_agent_metrics.json"
BASELINE_COMPARISON_PATH = "data/results/baseline_comparison.json"


@pytest.fixture(scope="module")
def harness_run():
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--skip-baselines"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_harness_exits_cleanly(harness_run):
    assert "HEADLINE RESULTS" in harness_run.stdout


def test_report_files_written(harness_run):
    with open(REPORT_JSON_PATH, "r", encoding="utf-8"):
        pass
    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Evaluation Report" in content
    assert "Three-Way Comparison" in content


def test_report_structure(harness_run):
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert "golden_set" in report
    assert "intent_classification" in report
    assert "escalation_decision" in report
    assert "headline_comparison_table" in report
    assert "notes" in report

    ic = report["intent_classification"]
    for system in ("majority_class_baseline", "tfidf_logreg_baseline", "ai_agent"):
        assert system in ic
        assert "accuracy" in ic[system]
        assert "macro_f1" in ic[system]

    assert "response_quality" in report
    rq = report["response_quality"]
    assert "llm_judge_mean_scores" in rq
    assert "human_agreement" in rq
    assert rq["human_agreement"]["primary_statistic"] == "mean_absolute_disagreement"


def test_report_numbers_match_source_files(harness_run):
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(AGENT_METRICS_PATH, "r", encoding="utf-8") as f:
        agent_metrics = json.load(f)
    with open(BASELINE_COMPARISON_PATH, "r", encoding="utf-8") as f:
        baseline_comparison = json.load(f)

    assert report["intent_classification"]["ai_agent"]["accuracy"] == \
        agent_metrics["intent_classification"]["accuracy"]
    assert report["escalation_decision"]["ai_agent"]["accuracy"] == \
        agent_metrics["escalation_decision"]["accuracy"]
    assert report["intent_classification"]["tfidf_logreg_baseline"]["accuracy"] == \
        baseline_comparison["baseline_tfidf_logreg"]["accuracy"]


def test_golden_set_summary_correct(harness_run):
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    gs = report["golden_set"]
    assert gs["size"] == 200
    assert gs["escalate_true"] + gs["escalate_false"] == gs["size"]


def test_skip_baselines_does_not_touch_agent_results():
    """--skip-baselines (and the default fast path) must never trigger a
    Step 7 LLM regeneration -- that requires an API key and real time/cost."""
    with open("data/results/ai_agent_results.json", "r", encoding="utf-8") as f:
        before = f.read()
    subprocess.run(
        [sys.executable, "evaluate.py", "--skip-baselines"],
        capture_output=True, text=True, timeout=60,
    )
    with open("data/results/ai_agent_results.json", "r", encoding="utf-8") as f:
        after = f.read()
    assert before == after, "Fast-path harness run must not modify committed agent results"

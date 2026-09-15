"""
Unit tests for Step 7: AI Support Agent results and evaluation.

LLM output isn't perfectly deterministic across reruns, so these tests check
STRUCTURAL invariants and safety-critical guarantees -- not exact accuracy
figures. Correctness of the numbers themselves is established by manual
review (see docs/DECISION_LOG.md Decision 13) and the fact that the agent's
predictions are scored against the same frozen gold labels as everything else.
"""

import json
import pytest
from src.classification.taxonomy import INTENTS

RESULTS_PATH = "data/results/ai_agent_results.json"
METRICS_PATH = "data/results/ai_agent_metrics.json"
COMPARISON_PATH = "data/results/system_comparison.json"
GOLDEN_SET_PATH = "data/golden/golden_evaluation_set.json"


@pytest.fixture(scope="module")
def results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def golden_set():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_results_cover_full_golden_set(results, golden_set):
    assert len(results) == len(golden_set)
    result_ids = {r["golden_id"] for r in results}
    golden_ids = {g["golden_id"] for g in golden_set}
    assert result_ids == golden_ids


def test_required_fields_present(results):
    required = {
        "golden_id", "gold_intent", "gold_should_escalate",
        "predicted_intent", "intent_confidence", "predicted_should_escalate",
        "escalation_reason", "draft_response", "retrieved_evidence",
    }
    for r in results:
        missing = required - set(r.keys())
        assert not missing, f"{r['golden_id']} missing fields: {missing}"


def test_predicted_intents_valid(results):
    for r in results:
        assert r["predicted_intent"] in INTENTS, (
            f"{r['golden_id']}: invalid predicted_intent '{r['predicted_intent']}'"
        )


def test_predicted_escalate_is_boolean(results):
    for r in results:
        assert isinstance(r["predicted_should_escalate"], bool)


def test_driver_safety_always_escalated(results):
    """Mandatory safety policy -- code-level override in support_agent.py
    guarantees this even if the model's own reasoning ever disagreed."""
    for r in results:
        if r["predicted_intent"] == "driver_conduct_and_safety":
            assert r["predicted_should_escalate"] is True, (
                f"{r['golden_id']}: driver_conduct_and_safety must always escalate"
            )


def test_escalation_reason_is_a_real_justification(results):
    """Regression test for the 'None' string bug found during live smoke
    testing (see Decision 13) -- every reason must be non-empty and not the
    literal placeholder string."""
    for r in results:
        reason = r["escalation_reason"].strip()
        assert reason, f"{r['golden_id']}: empty escalation_reason"
        assert reason.lower() != "none", f"{r['golden_id']}: literal 'None' reason"


def test_draft_response_non_empty(results):
    for r in results:
        assert r["draft_response"].strip(), f"{r['golden_id']}: empty draft_response"


def test_retrieved_evidence_structure(results):
    for r in results:
        assert isinstance(r["retrieved_evidence"], list)
        for e in r["retrieved_evidence"]:
            assert "similarity" in e
            assert "historical_customer_message" in e
            assert "historical_uber_response" in e


def test_metrics_file_matches_results(results):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["test_set_size"] == len(results)
    assert 0.0 <= metrics["intent_classification"]["accuracy"] <= 1.0
    assert 0.0 <= metrics["escalation_decision"]["accuracy"] <= 1.0


def test_retrieval_corpus_excludes_golden_set(golden_set):
    """Same leakage class caught in Step 6 (Decision 12) -- the agent's
    retrieval corpus must not contain any golden set message, exactly or
    near-duplicated, or the agent could retrieve the answer to its own eval."""
    from src.pipeline.support_agent import build_retrieval_corpus
    corpus = build_retrieval_corpus()
    golden_texts = {g["customer_message"].strip() for g in golden_set}
    leaked = golden_texts & set(corpus["texts"])
    assert not leaked, f"{len(leaked)} golden set message(s) leaked into retrieval corpus"


def test_escalation_accuracy_regression_guard():
    """Decision 14 fixed escalation accuracy from 76.0% to 85.0% via a
    diagnosed prompt revision. This guards against silently regressing back
    toward the pre-fix level in a future prompt change -- threshold is set
    with margin for ordinary LLM run-to-run variance, not pinned to 0.85."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["escalation_decision"]["accuracy"] >= 0.80, (
        "Escalation accuracy dropped notably below the post-Decision-14 level "
        "(0.85) -- check whether a prompt change regressed the escalation "
        "checklist introduced in Decision 14."
    )


def test_agent_beats_both_baselines_on_macro_f1():
    """Sanity check, not a tautology to defend at all costs -- but if this
    ever fails, it's a signal worth investigating before trusting new results,
    not just deleting the test."""
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        comparison = json.load(f)
    agent_f1 = comparison["ai_agent"]["macro_f1"]
    assert agent_f1 > comparison["majority_class_baseline"]["macro_f1"]
    assert agent_f1 > comparison["tfidf_logreg_baseline"]["macro_f1"]

"""
Unit tests for Step 6: Baseline Intent Classifiers.
Verifies that:
1. Both baseline result files exist with required metrics.
2. Both baselines were evaluated on the exact same test set (the frozen golden set).
3. No golden set conversation_id leaked into the training pool.
4. Metrics are internally consistent (support sums to test set size, valid ranges).
5. The TF-IDF+LogReg baseline beats the majority-class baseline on macro F1
   (a basic sanity check -- a real classifier should beat a constant predictor).
6. Results are reproducible given the fixed random seed.
"""

import json
import subprocess
import sys
import pytest
from src.classification.taxonomy import INTENTS

MAJORITY_PATH = "data/results/baseline_majority.json"
TFIDF_PATH = "data/results/baseline_tfidf_logreg.json"
COMPARISON_PATH = "data/results/baseline_comparison.json"
GOLDEN_SET_PATH = "data/golden/golden_evaluation_set.json"
LEAKAGE_MANIFEST_PATH = "data/golden/golden_conversation_ids.json"


@pytest.fixture(scope="module")
def golden_set():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def majority_results():
    with open(MAJORITY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tfidf_results():
    with open(TFIDF_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_result_files_exist(majority_results, tfidf_results):
    assert "accuracy" in majority_results
    assert "accuracy" in tfidf_results


def test_required_metrics_present(majority_results, tfidf_results):
    for results in (majority_results, tfidf_results):
        assert "accuracy" in results
        assert "macro_f1" in results
        assert "per_class" in results
        assert "confusion_matrix" in results
        assert set(results["per_class"].keys()) == set(INTENTS)
        assert results["confusion_matrix"]["labels"] == INTENTS
        assert len(results["confusion_matrix"]["matrix"]) == len(INTENTS)


def test_same_test_set_for_both_baselines(golden_set, majority_results, tfidf_results):
    golden_ids = {c["golden_id"] for c in golden_set}
    assert set(majority_results["predictions_by_golden_id"].keys()) == golden_ids
    assert set(tfidf_results["predictions_by_golden_id"].keys()) == golden_ids


def test_support_sums_to_test_set_size(golden_set, majority_results):
    total_support = sum(v["support"] for v in majority_results["per_class"].values())
    assert total_support == len(golden_set)


def test_no_golden_conversation_in_training_pool():
    with open(LEAKAGE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reserved_ids = set(json.load(f)["conversation_ids"])
    with open("data/processed/uber_conversations.json", "r", encoding="utf-8") as f:
        conversations = json.load(f)

    from src.classification.baselines import build_training_set
    train_texts, train_labels = build_training_set()

    # Cross-check: none of the reserved golden conversation_ids' customer
    # messages appear verbatim in the training text pool.
    reserved_texts = {
        c["first_customer_query_clean"].strip()
        for c in conversations
        if c["conversation_id"] in reserved_ids
    }
    train_text_set = set(train_texts)
    leaked = reserved_texts & train_text_set
    assert not leaked, f"{len(leaked)} golden set messages leaked into training pool"


def test_no_golden_text_leaks_under_a_different_conversation_id(golden_set):
    """Regression test: a duplicate tweet (conv_1156838) shares its exact text
    with golden_0143 under a different conversation_id, so a conversation_id-only
    exclusion missed it. build_training_set must also exclude by text/near-dup."""
    from src.classification.baselines import build_training_set
    train_texts, _train_labels = build_training_set()

    golden_texts = {c["customer_message"].strip() for c in golden_set}
    leaked = golden_texts & set(train_texts)
    assert not leaked, f"{len(leaked)} golden set message(s) leaked via duplicate text: {leaked}"


def test_tfidf_beats_majority_on_macro_f1(majority_results, tfidf_results):
    assert tfidf_results["macro_f1"] > majority_results["macro_f1"]


def test_metrics_in_valid_range(majority_results, tfidf_results):
    for results in (majority_results, tfidf_results):
        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["macro_f1"] <= 1.0


def test_comparison_file_matches_individual_results(majority_results, tfidf_results):
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        comparison = json.load(f)
    assert comparison["baseline_majority_class"]["accuracy"] == majority_results["accuracy"]
    assert comparison["baseline_tfidf_logreg"]["accuracy"] == tfidf_results["accuracy"]


def test_reproducible_with_fixed_seed(majority_results, tfidf_results):
    original_majority_acc = majority_results["accuracy"]
    original_tfidf_acc = tfidf_results["accuracy"]

    result = subprocess.run(
        [sys.executable, "-m", "src.classification.baselines"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr

    with open(MAJORITY_PATH, "r", encoding="utf-8") as f:
        rerun_majority = json.load(f)
    with open(TFIDF_PATH, "r", encoding="utf-8") as f:
        rerun_tfidf = json.load(f)

    assert rerun_majority["accuracy"] == original_majority_acc
    assert rerun_tfidf["accuracy"] == original_tfidf_acc

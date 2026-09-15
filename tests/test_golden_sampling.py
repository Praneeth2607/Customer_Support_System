"""
Unit tests for Step 5: Golden Evaluation Set sampling.
Verifies that:
1. The candidate pool, leakage manifest, and summary files exist.
2. The pool size is within the 150-250 range required by the assignment.
3. Every candidate has the required schema fields and a unique golden_id/conversation_id.
4. The leakage manifest's conversation_ids exactly match the candidate pool's.
5. gold_intent / should_escalate are still null (sampling must not silently pre-fill labels).
6. The sample is reproducible: re-running the script with the same seed yields the same conversation_ids.
"""

import json
import subprocess
import sys
import pytest

CANDIDATES_PATH = "data/golden/golden_candidates.json"
MANIFEST_PATH = "data/golden/golden_conversation_ids.json"
SUMMARY_PATH = "data/golden/golden_sampling_summary.json"

REQUIRED_FIELDS = {
    "golden_id", "conversation_id", "customer_message", "customer_message_raw",
    "num_turns", "conversation_context", "historical_resolution",
    "pseudo_intent_heuristic", "matched_intent_keywords", "difficulty_category",
    "sampling_category", "gold_intent", "should_escalate", "escalation_reason",
    "predicted_intent", "labeling_notes",
}


@pytest.fixture(scope="module")
def candidates():
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_output_files_exist():
    for path in (CANDIDATES_PATH, MANIFEST_PATH, SUMMARY_PATH):
        with open(path, "r", encoding="utf-8"):
            pass


def test_pool_size_in_range(candidates):
    assert 150 <= len(candidates) <= 250, (
        f"Golden set must have 150-250 examples, got {len(candidates)}"
    )


def test_schema_fields_present(candidates):
    for c in candidates:
        missing = REQUIRED_FIELDS - set(c.keys())
        assert not missing, f"Candidate {c.get('golden_id')} missing fields: {missing}"


def test_unique_ids(candidates):
    golden_ids = [c["golden_id"] for c in candidates]
    conv_ids = [c["conversation_id"] for c in candidates]
    assert len(golden_ids) == len(set(golden_ids)), "Duplicate golden_id found"
    assert len(conv_ids) == len(set(conv_ids)), "Duplicate conversation_id found -- leakage risk"


def test_labels_not_prefilled(candidates):
    for c in candidates:
        assert c["gold_intent"] is None, (
            "gold_intent must stay null until the hand-labelling pass runs"
        )
        assert c["should_escalate"] is None, (
            "should_escalate must stay null until the hand-labelling pass runs"
        )


def test_leakage_manifest_matches_candidates(candidates):
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_ids = set(manifest["conversation_ids"])
    candidate_ids = {c["conversation_id"] for c in candidates}
    assert manifest_ids == candidate_ids, (
        "Leakage manifest must exactly match the candidate pool's conversation_ids"
    )
    assert manifest["count"] == len(candidates)


def test_all_seven_intents_represented(candidates):
    from src.classification.taxonomy import INTENTS
    pseudo_intents = {c["pseudo_intent_heuristic"] for c in candidates}
    for intent in INTENTS:
        assert intent in pseudo_intents, f"No candidate sampled for {intent}"


def test_reproducible_with_fixed_seed(candidates):
    """Re-running the sampler must select the exact same conversation_ids."""
    original_ids = sorted(c["conversation_id"] for c in candidates)

    result = subprocess.run(
        [sys.executable, "src/evaluation/sample_golden_set.py"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        rerun_candidates = json.load(f)
    rerun_ids = sorted(c["conversation_id"] for c in rerun_candidates)

    assert rerun_ids == original_ids, "Sampling is not deterministic across runs"

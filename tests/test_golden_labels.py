"""
Unit tests for Step 5b: hand-labelled, frozen golden evaluation set.
Verifies that:
1. data/golden/golden_evaluation_set.json exists and has 150-250 fully labelled examples.
2. Every gold_intent is a valid taxonomy intent; every should_escalate is boolean.
3. driver_conduct_and_safety examples are always escalated (mandatory policy, no exceptions).
4. No example is left with a null label (the set must be fully hand-labelled, not partial).
5. Every gold_intent is represented at least once (per Step 5 sampling design).
"""

import json
import pytest
from src.classification.taxonomy import INTENTS

GOLDEN_SET_PATH = "data/golden/golden_evaluation_set.json"


@pytest.fixture(scope="module")
def golden_set():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_golden_set_exists_and_sized(golden_set):
    assert 150 <= len(golden_set) <= 250


def test_all_labels_present(golden_set):
    for c in golden_set:
        assert c["gold_intent"] is not None, f"{c['golden_id']} missing gold_intent"
        assert c["should_escalate"] is not None, f"{c['golden_id']} missing should_escalate"
        assert c["escalation_reason"], f"{c['golden_id']} missing escalation_reason"
        assert c["labeling_notes"], f"{c['golden_id']} missing labeling_notes"


def test_gold_intents_valid(golden_set):
    for c in golden_set:
        assert c["gold_intent"] in INTENTS, (
            f"{c['golden_id']}: invalid gold_intent '{c['gold_intent']}'"
        )


def test_should_escalate_is_boolean(golden_set):
    for c in golden_set:
        assert isinstance(c["should_escalate"], bool)


def test_driver_safety_always_escalates(golden_set):
    """driver_conduct_and_safety has a MANDATORY escalate policy -- no exceptions allowed."""
    for c in golden_set:
        if c["gold_intent"] == "driver_conduct_and_safety":
            assert c["should_escalate"] is True, (
                f"{c['golden_id']}: driver_conduct_and_safety must always escalate"
            )


def test_all_intents_represented(golden_set):
    gold_intents = {c["gold_intent"] for c in golden_set}
    for intent in INTENTS:
        assert intent in gold_intents, f"No golden example labelled {intent}"


def test_unique_golden_and_conversation_ids(golden_set):
    golden_ids = [c["golden_id"] for c in golden_set]
    conv_ids = [c["conversation_id"] for c in golden_set]
    assert len(golden_ids) == len(set(golden_ids))
    assert len(conv_ids) == len(set(conv_ids))

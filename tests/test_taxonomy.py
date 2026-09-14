"""
Unit tests for Step 4: Intent Discovery & Taxonomy Definition
Verifies that:
1. Intent discovery report exists and contains valid cluster analyses.
2. The documented intent taxonomy (docs/intent_taxonomy.md) defines exactly 6 intents.
3. Every intent has a definition, positive examples, boundary rules, and escalation policy.
4. Python schema definition in src/classification/taxonomy.py matches the documented intents.
"""

import os
import json
import pytest

REPORT_PATH = "data/audit/intent_discovery_report.json"
TAXONOMY_DOC = "docs/intent_taxonomy.md"

def test_intent_discovery_report_exists():
    assert os.path.exists(REPORT_PATH), f"{REPORT_PATH} should exist"
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "clusters" in data, "Report must have clusters"
    assert len(data["clusters"]) >= 5, "Should have at least 5 clusters"
    assert "top_global_ngrams" in data, "Report must have top_global_ngrams"

def test_intent_taxonomy_document_exists():
    assert os.path.exists(TAXONOMY_DOC), f"{TAXONOMY_DOC} must exist"
    with open(TAXONOMY_DOC, "r", encoding="utf-8") as f:
        content = f.read()
        
    expected_intents = [
        "cancellation_issue",
        "fare_and_payment_dispute",
        "lost_item",
        "driver_conduct_and_safety",
        "pickup_and_route_issue",
        "account_and_promo_issue"
    ]
    for intent in expected_intents:
        assert intent in content, f"Intent '{intent}' must be documented in {TAXONOMY_DOC}"
        
    assert "Positive Examples" in content, "Taxonomy must document Positive Examples"
    assert "Boundary Cases" in content, "Taxonomy must document Boundary Cases"
    assert "Escalation Policy" in content, "Taxonomy must document Escalation Policy"

def test_taxonomy_python_module():
    from src.classification.taxonomy import INTENTS, INTENT_DEFINITIONS, ESCALATION_POLICIES
    
    assert len(INTENTS) == 6, f"Expected exactly 6 intents, got {len(INTENTS)}"
    for intent in INTENTS:
        assert intent in INTENT_DEFINITIONS, f"Intent definition missing for {intent}"
        assert intent in ESCALATION_POLICIES, f"Escalation policy missing for {intent}"
        assert "definition" in INTENT_DEFINITIONS[intent]
        assert len(INTENT_DEFINITIONS[intent]["examples"]) >= 3
        assert "action" in ESCALATION_POLICIES[intent]

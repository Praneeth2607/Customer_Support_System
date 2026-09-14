"""
Unit tests for Step 3: Conversation Thread Reconstruction & Text Normalization
Verifies dialogue reconstruction integrity, schema structure, turn ordering, and text cleaning.
"""

import os
import json
import pytest
from src.data.reconstruct_conversations import clean_tweet_text

CONVERSATIONS_JSON = "data/processed/uber_conversations.json"
SUMMARY_JSON = "data/processed/uber_conversations_summary.json"

def test_clean_tweet_text():
    # Test HTML unescaping
    assert clean_tweet_text("Driver was rude &amp; late") == "Driver was rude & late"
    
    # Test URL replacement
    assert clean_tweet_text("Look at this link https://t.co/abc123xyz now") == "Look at this link [URL] now"
    
    # Test leading @mentions removal
    assert clean_tweet_text("@Uber_Support @105836 My driver never arrived") == "My driver never arrived"
    
    # Test whitespace normalization
    assert clean_tweet_text("Too    many \n\n spaces   here") == "Too many spaces here"

def test_reconstructed_files_exist():
    assert os.path.exists(CONVERSATIONS_JSON), f"{CONVERSATIONS_JSON} should exist"
    assert os.path.getsize(CONVERSATIONS_JSON) > 1_000_000, "Conversations JSON should be >1MB"
    assert os.path.exists(SUMMARY_JSON), f"{SUMMARY_JSON} should exist"

def test_conversations_schema_and_integrity():
    with open(CONVERSATIONS_JSON, "r", encoding="utf-8") as f:
        # Load sample of 1000 conversations to keep test fast
        data = json.load(f)
        
    assert len(data) >= 25_000, f"Expected at least 25,000 conversations, got {len(data)}"
    
    sample = data[:100]
    required_keys = [
        "conversation_id", "root_tweet_id", "customer_id", "created_at",
        "num_turns", "first_customer_query", "first_customer_query_clean",
        "first_brand_response", "first_brand_response_clean",
        "has_dm_deflection", "has_help_url", "messages"
    ]
    
    for conv in sample:
        for k in required_keys:
            assert k in conv, f"Missing key '{k}' in conversation {conv.get('conversation_id')}"
            
        assert conv["num_turns"] >= 2, "Every conversation must have at least 2 turns (Customer + Brand)"
        assert len(conv["first_customer_query_clean"]) > 0, "Customer query must not be empty"
        assert len(conv["first_brand_response_clean"]) > 0, "Brand response must not be empty"
        
        # Verify first message is customer, second is brand
        assert conv["messages"][0]["speaker"] == "customer", "First turn must be customer"
        assert any(m["speaker"] == "uber" for m in conv["messages"]), "Must contain an Uber response"

def test_summary_metrics():
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert summary["total_conversations"] >= 25_000
    assert summary["two_turn_pairs"] > 0
    assert summary["avg_turns_per_conversation"] >= 2.0
    assert summary["avg_customer_query_words"] > 5.0

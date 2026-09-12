"""
Unit tests for Step 2: Uber Data Extraction & Conversation Audit
Verifies extraction outputs, schema conformity, thread linkage metrics, and data integrity.
"""

import os
import json
import pandas as pd
import pytest

PROCESSED_CSV = "data/processed/uber_tweets.csv"
SUMMARY_JSON = "data/processed/uber_audit_summary.json"

def test_extracted_csv_exists():
    assert os.path.exists(PROCESSED_CSV), f"{PROCESSED_CSV} should exist after extraction"
    assert os.path.getsize(PROCESSED_CSV) > 1_000_000, "Extracted CSV should be larger than 1MB"

def test_extracted_summary_json_exists():
    assert os.path.exists(SUMMARY_JSON), f"{SUMMARY_JSON} should exist after extraction"

def test_extraction_schema_and_integrity():
    df = pd.read_csv(PROCESSED_CSV, nrows=500)
    expected_cols = [
        'tweet_id', 'author_id', 'inbound', 'created_at',
        'text', 'response_tweet_id', 'in_response_to_tweet_id'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' missing from extracted Uber CSV"
        
    # Check that outbound authors are Uber_Support
    outbound = df[~df['inbound'].astype(bool)]
    if not outbound.empty:
        assert (outbound['author_id'] == 'Uber_Support').all(), "All outbound tweets must be authored by Uber_Support"

def test_conversation_metrics_validity():
    with open(SUMMARY_JSON, "r") as f:
        summary = json.load(f)
        
    required_keys = [
        "total_extracted_tweets", "customer_inbound_tweets", "uber_outbound_responses",
        "unique_customers", "total_conversation_threads", "usable_conversations_with_replies",
        "multi_turn_conversations_3plus", "thread_length_mean", "customer_text_duplicate_rate_pct",
        "uber_response_duplicate_rate_pct", "top_5_uber_response_templates"
    ]
    for k in required_keys:
        assert k in summary, f"Key '{k}' missing from summary JSON"
        
    assert summary["uber_outbound_responses"] >= 50_000, "Should have at least 50k Uber responses"
    assert summary["customer_inbound_tweets"] >= 20_000, "Should have at least 20k inbound customer tweets"
    assert summary["usable_conversations_with_replies"] >= 15_000, "Should have at least 15k paired conversations"
    assert summary["thread_length_mean"] >= 1.5, "Average thread length should be >= 1.5"

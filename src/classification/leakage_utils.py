"""
Shared leakage-safe corpus access.

Both src/classification/baselines.py (Step 6) and src/pipeline/support_agent.py
(Step 7's retrieval corpus) need the full ~42k conversation corpus with the
golden evaluation set excluded. Centralizing the exclusion logic here means
the leakage protection is tested once and reused everywhere, instead of being
reimplemented (and potentially reimplemented incorrectly) per consumer.

Excludes by BOTH conversation_id (the leakage manifest) AND exact/near-duplicate
customer-message text -- conversation_id alone missed a real leak during Step 6
(see Decision 12: a duplicate tweet, conv_1156838, shared exact text with
golden_0143 under a different conversation_id).
"""

import json
from src.evaluation.sample_golden_set import near_duplicate_signature

CONVERSATIONS_PATH = "data/processed/uber_conversations.json"
GOLDEN_SET_PATH = "data/golden/golden_evaluation_set.json"
LEAKAGE_MANIFEST_PATH = "data/golden/golden_conversation_ids.json"


def iter_leakage_safe_conversations():
    """Yield conversations from the full corpus, excluding anything reserved
    for or duplicated from the golden evaluation set. Also applies the same
    len>5 substantive-text filter used throughout Steps 4-6."""
    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    with open(LEAKAGE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reserved_ids = set(json.load(f)["conversation_ids"])
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    golden_texts = {c["customer_message"].strip() for c in golden_set}
    golden_signatures = {near_duplicate_signature(c["customer_message"]) for c in golden_set}

    for conv in conversations:
        if conv["conversation_id"] in reserved_ids:
            continue
        text = conv["first_customer_query_clean"].strip()
        if len(text) <= 5:
            continue
        if text in golden_texts or near_duplicate_signature(text) in golden_signatures:
            continue
        yield conv

"""
Step 3: Conversation Thread Reconstruction & Dialogue Normalization

Takes raw extracted Uber tweets from data/processed/uber_tweets.csv and:
1. Reconstructs full conversation trees from customer root tweets down through brand replies and follow-ups.
2. Orders messages chronologically / topologically (Customer -> Uber -> Customer -> Uber).
3. Applies text normalization:
   - Unescapes HTML entities (&amp;, &lt;, &gt;)
   - Normalizes/strips leading @mentions without losing conversational context
   - Replaces t.co shortlinks with standardized [URL] tokens
   - Normalizes whitespace
4. Enriches each conversation object with metadata:
   - conversation_id (root tweet id)
   - customer_id
   - num_turns
   - first_customer_query (raw + clean)
   - first_brand_response (raw + clean)
   - full dialogue messages array
   - has_dm_deflection (boolean)
   - has_help_url (boolean)
5. Saves clean structured conversations to data/processed/uber_conversations.json
"""

import os
import re
import html
import json
import time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

def clean_tweet_text(text: str) -> str:
    """
    Cleans and normalizes tweet text for ML models and LLM prompts:
    - HTML entity unescaping (&amp; -> &)
    - Replaces t.co URLs with [URL] token
    - Removes leading brand/user handles (@Uber_Support, @115872) while preserving in-sentence mentions
    - Strips excess whitespace and newlines
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Unescape HTML entities
    cleaned = html.unescape(text)
    
    # 2. Replace URLs with [URL]
    cleaned = re.sub(r'https?://t\.co/\w+', '[URL]', cleaned)
    cleaned = re.sub(r'https?://\S+', '[URL]', cleaned)
    
    # 3. Strip leading @mentions (e.g., "@Uber_Support @105836 hi there" -> "hi there")
    cleaned = re.sub(r'^(?:@\w+\s*)+', '', cleaned)
    
    # 4. Remove residual carriage returns and normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def parse_twitter_date(date_series: pd.Series) -> pd.Series:
    """Parses Twitter RFC 2822 timestamps: 'Wed Oct 11 06:55:44 +0000 2017'"""
    return pd.to_datetime(date_series, format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')

def reconstruct_conversations(
    input_csv="data/processed/uber_tweets.csv",
    output_json="data/processed/uber_conversations.json",
    summary_json="data/processed/uber_conversations_summary.json",
    max_turns=6
):
    print("=" * 70)
    print("STEP 3: CONVERSATION RECONSTRUCTION & DIALOGUE NORMALIZATION")
    print("=" * 70)
    
    start_time = time.time()
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df):,} tweets from {input_csv} ({time.time() - start_time:.2f}s)")
    
    # Parse timestamps for exact chronological sequencing
    df['parsed_date'] = parse_twitter_date(df['created_at'])
    
    # Index tweets by tweet_id
    tweet_dict = {}
    parent_to_children = defaultdict(list)
    tweet_to_parent = {}
    
    for _, row in df.iterrows():
        t_id = int(row['tweet_id'])
        inbound = bool(row['inbound'])
        author = str(row['author_id'])
        text = str(row['text']) if pd.notnull(row['text']) else ""
        date = row['parsed_date']
        
        tweet_dict[t_id] = {
            "tweet_id": t_id,
            "speaker": "customer" if inbound else "uber",
            "author_id": author,
            "created_at": str(row['created_at']),
            "parsed_date": date,
            "text": text,
            "text_clean": clean_tweet_text(text),
            "in_response_to": None
        }
        
        if pd.notnull(row['in_response_to_tweet_id']):
            try:
                p_id = int(row['in_response_to_tweet_id'])
                tweet_dict[t_id]["in_response_to"] = p_id
                tweet_to_parent[t_id] = p_id
                parent_to_children[p_id].append(t_id)
            except (ValueError, TypeError):
                pass

    print(f"Indexed {len(tweet_dict):,} tweets into graph structures.")
    
    # Identify conversation roots
    # A conversation root is a tweet whose parent is not in our dataset (or is null)
    # AND it must be an inbound customer tweet that initiated an issue
    root_ids = [
        t_id for t_id, data in tweet_dict.items()
        if (data["in_response_to"] is None or data["in_response_to"] not in tweet_dict)
        and data["speaker"] == "customer"
    ]
    print(f"Identified {len(root_ids):,} customer root tweets.")
    
    dm_regex = re.compile(r'\b(dm|direct message|private message)\b', re.IGNORECASE)
    
    conversations = []
    skipped_no_reply = 0
    skipped_empty_text = 0
    
    for root_id in root_ids:
        # Traverse path: from root, follow children.
        # Most conversations are a linear sequence: Customer -> Brand -> Customer -> Brand.
        # In case of branching, we follow the primary chronological path.
        path = [root_id]
        curr = root_id
        
        while len(path) < max_turns:
            children = parent_to_children.get(curr, [])
            if not children:
                break
            # Filter children present in tweet_dict and sort chronologically
            valid_children = [c for c in children if c in tweet_dict]
            if not valid_children:
                break
            # Pick the earliest direct response
            valid_children.sort(key=lambda x: tweet_dict[x]["parsed_date"] if pd.notnull(tweet_dict[x]["parsed_date"]) else 0)
            next_tweet = valid_children[0]
            path.append(next_tweet)
            curr = next_tweet
            
        # Verify that the conversation has at least 1 customer tweet AND at least 1 Uber response
        speakers = [tweet_dict[t]["speaker"] for t in path]
        if "uber" not in speakers:
            skipped_no_reply += 1
            continue
            
        # Extract initial customer query and first brand reply
        first_cust = tweet_dict[path[0]]
        first_brand = None
        for t in path:
            if tweet_dict[t]["speaker"] == "uber":
                first_brand = tweet_dict[t]
                break
                
        # Quality filter: discard conversations with empty or low-signal text (e.g. only mentions)
        if not first_cust["text_clean"].strip() or not first_brand or not first_brand["text_clean"].strip():
            skipped_empty_text += 1
            continue
                
        # Flags
        brand_texts = " ".join([tweet_dict[t]["text"] for t in path if tweet_dict[t]["speaker"] == "uber"])
        has_dm = bool(dm_regex.search(brand_texts))
        has_url = "[URL]" in brand_texts or "http" in brand_texts
        
        # Build message history
        messages = []
        for turn_idx, t in enumerate(path, 1):
            t_data = tweet_dict[t]
            messages.append({
                "turn": turn_idx,
                "tweet_id": t_data["tweet_id"],
                "speaker": t_data["speaker"],
                "author_id": t_data["author_id"],
                "created_at": t_data["created_at"],
                "text": t_data["text"],
                "text_clean": t_data["text_clean"]
            })
            
        conv_obj = {
            "conversation_id": f"conv_{root_id}",
            "root_tweet_id": root_id,
            "customer_id": first_cust["author_id"],
            "created_at": first_cust["created_at"],
            "num_turns": len(messages),
            "first_customer_query": first_cust["text"],
            "first_customer_query_clean": first_cust["text_clean"],
            "first_brand_response": first_brand["text"] if first_brand else "",
            "first_brand_response_clean": first_brand["text_clean"] if first_brand else "",
            "has_dm_deflection": has_dm,
            "has_help_url": has_url,
            "messages": messages
        }
        conversations.append(conv_obj)
        
    print(f"\nReconstructed {len(conversations):,} complete conversations (skipped {skipped_no_reply:,} without reply, {skipped_empty_text:,} empty text).")
    
    # Save conversations JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
        
    file_size_mb = os.path.getsize(output_json) / (1024 * 1024)
    print(f"Saved {len(conversations):,} conversations to {output_json} ({file_size_mb:.2f} MB)")
    
    # Compute summary statistics
    turn_counts = [c["num_turns"] for c in conversations]
    query_lens = [len(c["first_customer_query_clean"].split()) for c in conversations]
    
    turn_dist = Counter(turn_counts)
    dm_count = sum(1 for c in conversations if c["has_dm_deflection"])
    url_count = sum(1 for c in conversations if c["has_help_url"])
    
    summary = {
        "total_conversations": len(conversations),
        "skipped_empty_text": skipped_empty_text,
        "total_messages_reconstructed": sum(turn_counts),
        "turn_distribution": {str(k): v for k, v in sorted(turn_dist.items())},
        "two_turn_pairs": turn_dist.get(2, 0),
        "multi_turn_conversations_3plus": sum(v for k, v in turn_dist.items() if k >= 3),
        "avg_turns_per_conversation": round(float(np.mean(turn_counts)), 2),
        "median_turns_per_conversation": float(np.median(turn_counts)),
        "max_turns_capped": max_turns,
        "avg_customer_query_words": round(float(np.mean(query_lens)), 2),
        "dm_deflection_rate_pct": round(dm_count / len(conversations) * 100, 2),
        "help_url_presence_rate_pct": round(url_count / len(conversations) * 100, 2),
        "sample_conversation": conversations[0] if conversations else {}
    }
    
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved reconstruction summary to {summary_json}")
    
    print("\n" + "=" * 60)
    print("CONVERSATION RECONSTRUCTION SUMMARY:")
    print("=" * 60)
    print(f"Total Reconstructed Conversations:  {summary['total_conversations']:,}")
    print(f"  - 2-Turn Dialogues (Cust -> Uber): {summary['two_turn_pairs']:,} ({summary['two_turn_pairs']/len(conversations)*100:.1f}%)")
    print(f"  - Multi-Turn Dialogues (>=3):      {summary['multi_turn_conversations_3plus']:,} ({summary['multi_turn_conversations_3plus']/len(conversations)*100:.1f}%)")
    print(f"Average Turns per Conversation:     {summary['avg_turns_per_conversation']}")
    print(f"Average Customer Query Length:      {summary['avg_customer_query_words']} words")
    print(f"DM Deflection Rate in Dialogues:    {summary['dm_deflection_rate_pct']}%")
    print(f"Help URL Rate in Dialogues:         {summary['help_url_presence_rate_pct']}%")
    
    return summary

if __name__ == "__main__":
    reconstruct_conversations()

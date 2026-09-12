"""
Step 2: Uber Support Brand Extraction & Conversation Graph Audit

Extracts all Uber-related tweets from twcs.csv using a two-pass graph-aware collection:
- Pass 1: Identify all Uber outbound tweets, their parent customer tweets (in_response_to),
  their child customer tweets (response_tweet_id), and inbound tweets directed to Uber.
- Pass 2: Stream and extract full metadata for all identified tweets into data/processed/uber_tweets.csv.

Computes and saves comprehensive conversational metrics:
- Total Uber tweets, customer tweets, Uber responses
- Conversation thread count, average/median/max thread length
- Complete vs incomplete thread ratios
- Duplicate text rates
- Number of usable customer-brand interaction pairs
"""

import os
import re
import json
import time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

def extract_uber_data(input_csv="twcs.csv", output_csv="data/processed/uber_tweets.csv", chunk_size=250_000):
    print("=" * 70)
    print("STEP 2: UBER SUPPORT EXTRACTION & CONVERSATION GRAPH AUDIT")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    start_time = time.time()
    
    # -------------------------------------------------------------
    # PASS 1: Identify all Uber-related tweet IDs
    # -------------------------------------------------------------
    print("\n[Pass 1/2] Scanning twcs.csv to map Uber conversation tweet IDs...")
    
    uber_tweet_ids = set()           # All outbound tweets by Uber
    linked_parent_ids = set()        # Customer tweets replied to by Uber
    linked_child_ids = set()         # Follow-up tweets responding to Uber
    direct_mention_ids = set()       # Inbound tweets explicitly mentioning @Uber_Support
    
    chunks_scanned = 0
    total_raw_rows = 0
    
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        chunks_scanned += 1
        total_raw_rows += len(chunk)
        
        # 1. Outbound Uber tweets
        uber_mask = (chunk['author_id'] == 'Uber_Support') & (~chunk['inbound'].astype(bool))
        uber_rows = chunk[uber_mask]
        
        for _, row in uber_rows.iterrows():
            t_id = row['tweet_id']
            uber_tweet_ids.add(t_id)
            
            # Record parent (the customer query being answered)
            if pd.notnull(row['in_response_to_tweet_id']):
                try:
                    linked_parent_ids.add(int(row['in_response_to_tweet_id']))
                except (ValueError, TypeError):
                    pass
                    
            # Record children (follow-ups)
            if pd.notnull(row['response_tweet_id']):
                # response_tweet_id can be comma-separated
                for resp_id_str in str(row['response_tweet_id']).split(','):
                    resp_id_str = resp_id_str.strip()
                    if resp_id_str.isdigit():
                        linked_child_ids.add(int(resp_id_str))
                        
        # 2. Inbound mentions of @Uber_Support
        inbound_chunk = chunk[chunk['inbound'].astype(bool)]
        mention_mask = inbound_chunk['text'].str.contains(r'@Uber_Support\b', case=False, na=False)
        for t_id in inbound_chunk[mention_mask]['tweet_id']:
            direct_mention_ids.add(int(t_id))
            
        print(f"  Scanned chunk {chunks_scanned} ({total_raw_rows:,} rows)... found {len(uber_tweet_ids):,} Uber tweets")
        
    all_target_ids = uber_tweet_ids | linked_parent_ids | linked_child_ids | direct_mention_ids
    print(f"\nPass 1 Complete in {time.time() - start_time:.1f}s:")
    print(f"  - Outbound Uber Tweets:     {len(uber_tweet_ids):,}")
    print(f"  - Linked Parent Tweets:     {len(linked_parent_ids):,}")
    print(f"  - Linked Child Tweets:      {len(linked_child_ids):,}")
    print(f"  - Direct Mention Inbound:   {len(direct_mention_ids):,}")
    print(f"  - Total Unique Target IDs:  {len(all_target_ids):,}")
    
    # -------------------------------------------------------------
    # PASS 2: Extract rows matching target IDs
    # -------------------------------------------------------------
    print(f"\n[Pass 2/2] Extracting full records to {output_csv}...")
    pass2_start = time.time()
    
    extracted_dfs = []
    
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        # Convert tweet_id to numeric for fast set lookup
        match_mask = chunk['tweet_id'].isin(all_target_ids)
        matched_rows = chunk[match_mask]
        if not matched_rows.empty:
            extracted_dfs.append(matched_rows)
            
    uber_df = pd.concat(extracted_dfs, ignore_index=True).drop_duplicates(subset=['tweet_id'])
    # Sort by created_at where possible or tweet_id
    uber_df.to_csv(output_csv, index=False)
    
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print(f"Pass 2 Complete in {time.time() - pass2_start:.1f}s!")
    print(f"Saved {len(uber_df):,} tweets to {output_csv} ({file_size_mb:.2f} MB)")
    
    # -------------------------------------------------------------
    # CONVERSATION GRAPH AUDIT & METRICS
    # -------------------------------------------------------------
    print("\nComputing Conversation Graph & Interaction Metrics...")
    audit_metrics = compute_uber_metrics(uber_df)
    
    summary_path = "data/processed/uber_audit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(audit_metrics, f, indent=2)
    print(f"Saved metrics summary to {summary_path}")
    
    return audit_metrics

def compute_uber_metrics(df: pd.DataFrame) -> dict:
    total_tweets = len(df)
    customer_tweets = df[df['inbound'].astype(bool)]
    uber_responses = df[~df['inbound'].astype(bool)]
    
    num_customer_tweets = len(customer_tweets)
    num_uber_responses = len(uber_responses)
    unique_customers = customer_tweets['author_id'].nunique()
    
    # Reconstruct conversation threads using graph linkage
    # Map each tweet to its parent
    parent_map = {}
    children_map = defaultdict(list)
    tweet_speaker = {}
    
    for _, row in df.iterrows():
        t_id = int(row['tweet_id'])
        speaker = 'uber' if row['author_id'] == 'Uber_Support' else 'customer'
        tweet_speaker[t_id] = speaker
        
        if pd.notnull(row['in_response_to_tweet_id']):
            try:
                p_id = int(row['in_response_to_tweet_id'])
                parent_map[t_id] = p_id
                children_map[p_id].append(t_id)
            except (ValueError, TypeError):
                pass
                
    # Find root tweets (tweets whose parent is not in our dataset or has no parent)
    all_tweet_ids = set(df['tweet_id'].astype(int))
    root_tweet_ids = [t for t in all_tweet_ids if t not in parent_map or parent_map[t] not in all_tweet_ids]
    
    # Traverse threads from each root
    threads = []
    visited = set()
    
    for root in root_tweet_ids:
        if root in visited:
            continue
        # BFS/DFS to collect all tweets in this thread tree
        thread = []
        queue = [root]
        visited.add(root)
        
        while queue:
            curr = queue.pop(0)
            thread.append(curr)
            for child in children_map.get(curr, []):
                if child not in visited and child in all_tweet_ids:
                    visited.add(child)
                    queue.append(child)
                    
        threads.append(thread)
        
    thread_lengths = [len(t) for t in threads]
    
    # Conversation thread classification:
    # Usable conversation: contains at least 1 customer tweet AND at least 1 Uber response
    usable_threads = 0
    customer_only_threads = 0
    uber_only_threads = 0
    multi_turn_threads = 0 # 3 or more tweets
    
    for t in threads:
        speakers = {tweet_speaker[t_id] for t_id in t}
        if 'customer' in speakers and 'uber' in speakers:
            usable_threads += 1
            if len(t) >= 3:
                multi_turn_threads += 1
        elif 'customer' in speakers:
            customer_only_threads += 1
        else:
            uber_only_threads += 1
            
    # Duplicate Analysis
    cust_texts = customer_tweets['text'].str.strip().str.lower()
    uber_texts = uber_responses['text'].str.strip().str.lower()
    
    cust_dup_rate = round((num_customer_tweets - cust_texts.nunique()) / max(num_customer_tweets, 1) * 100, 2)
    uber_dup_rate = round((num_uber_responses - uber_texts.nunique()) / max(num_uber_responses, 1) * 100, 2)
    
    # Top repeated Uber response templates
    top_uber_templates = Counter(uber_texts).most_common(5)
    
    metrics = {
        "total_extracted_tweets": total_tweets,
        "customer_inbound_tweets": num_customer_tweets,
        "uber_outbound_responses": num_uber_responses,
        "unique_customers": unique_customers,
        "total_conversation_threads": len(threads),
        "usable_conversations_with_replies": usable_threads,
        "multi_turn_conversations_3plus": multi_turn_threads,
        "customer_unanswered_threads": customer_only_threads,
        "thread_length_mean": round(float(np.mean(thread_lengths)), 2) if thread_lengths else 0,
        "thread_length_median": float(np.median(thread_lengths)) if thread_lengths else 0,
        "thread_length_max": int(np.max(thread_lengths)) if thread_lengths else 0,
        "customer_text_duplicate_rate_pct": cust_dup_rate,
        "uber_response_duplicate_rate_pct": uber_dup_rate,
        "top_5_uber_response_templates": [
            {"template": t[:120] + "...", "frequency": c} for t, c in top_uber_templates
        ]
    }
    
    print("\n" + "=" * 60)
    print("CONVERSATION AUDIT SUMMARY:")
    print("=" * 60)
    print(f"Total Extracted Tweets:         {metrics['total_extracted_tweets']:,}")
    print(f"  - Customer Tweets:            {metrics['customer_inbound_tweets']:,}")
    print(f"  - Uber Outbound Tweets:       {metrics['uber_outbound_responses']:,}")
    print(f"  - Unique Customers:           {metrics['unique_customers']:,}")
    print(f"Total Conversation Threads:     {metrics['total_conversation_threads']:,}")
    print(f"  - Usable Paired Conversations:{metrics['usable_conversations_with_replies']:,}")
    print(f"  - Multi-Turn Threads (>=3):   {metrics['multi_turn_conversations_3plus']:,}")
    print(f"  - Unanswered Customer Threads:{metrics['customer_unanswered_threads']:,}")
    print(f"Thread Lengths (Mean/Med/Max):  {metrics['thread_length_mean']} / {metrics['thread_length_median']} / {metrics['thread_length_max']}")
    print(f"Customer Text Duplicate Rate:   {metrics['customer_text_duplicate_rate_pct']}%")
    print(f"Uber Response Template Dup Rate:{metrics['uber_response_duplicate_rate_pct']}%")
    
    return metrics

if __name__ == "__main__":
    extract_uber_data()

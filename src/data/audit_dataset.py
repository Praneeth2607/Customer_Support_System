"""
Dataset Audit & Schema Inspection Script
Analyzes twcs.csv and sample.csv using chunked processing to conserve memory.
Calculates:
- Total rows, columns, null counts, memory usage
- Inbound vs outbound distribution
- Top brands by volume
- Uber handle identification and volume statistics
- Conversation graph linkage statistics (roots, replies, branching)
"""

import os
import sys
import json
import time
from collections import Counter
import pandas as pd

def audit_sample(sample_path="sample.csv"):
    print("=" * 60)
    print(f"AUDITING SAMPLE FILE: {sample_path}")
    print("=" * 60)
    df = pd.read_csv(sample_path)
    print(f"Sample Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nData Types & Null Counts:")
    for col in df.columns:
        null_cnt = df[col].isnull().sum()
        print(f"  - {col:25s} | Type: {str(df[col].dtype):10s} | Nulls: {null_cnt} ({null_cnt/len(df)*100:.1f}%)")
    
    print("\nInbound Distribution in sample:")
    print(df['inbound'].value_counts(dropna=False).to_dict())
    
    brand_authors = df[~df['inbound']]['author_id'].value_counts().to_dict()
    print("\nBrand authors in sample (inbound=False):")
    for b, c in brand_authors.items():
        print(f"  {b}: {c}")

def audit_full_dataset(full_path="twcs.csv", chunk_size=200_000):
    print("\n" + "=" * 60)
    print(f"AUDITING FULL DATASET: {full_path}")
    print("=" * 60)
    
    file_size_bytes = os.path.getsize(full_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    
    start_time = time.time()
    
    total_rows = 0
    null_counts = Counter()
    inbound_counts = Counter()
    brand_counts = Counter()
    all_authors_count = 0
    unique_authors = set()
    
    # Linkage tracking
    has_in_response_to = 0
    has_response_tweet_id = 0
    is_root_tweet = 0
    
    # Uber specific
    uber_outbound_count = 0
    uber_text_mentions = 0
    
    # Check headers
    first_chunk = True
    col_names = []
    
    print(f"Processing in chunks of {chunk_size:,} rows...")
    for i, chunk in enumerate(pd.read_csv(full_path, chunksize=chunk_size, low_memory=False)):
        if first_chunk:
            col_names = list(chunk.columns)
            first_chunk = False
            
        chunk_len = len(chunk)
        total_rows += chunk_len
        
        # Null counts
        for col in chunk.columns:
            null_counts[col] += chunk[col].isnull().sum()
            
        # Inbound
        inbound_series = chunk['inbound'].astype(bool)
        inbound_counts[True] += inbound_series.sum()
        inbound_counts[False] += (~inbound_series).sum()
        
        # Brands (authors of outbound tweets)
        outbound_chunk = chunk[~inbound_series]
        for author in outbound_chunk['author_id'].dropna():
            brand_counts[author] += 1
            if 'uber' in str(author).lower():
                uber_outbound_count += 1
                
        # Uber text mentions (inbound or outbound mentioning uber)
        # Fast vectorized string check
        uber_mention_mask = chunk['text'].str.contains('uber', case=False, na=False)
        uber_text_mentions += uber_mention_mask.sum()
        
        # Linkages
        in_resp = chunk['in_response_to_tweet_id'].notnull()
        has_in_response_to += in_resp.sum()
        is_root_tweet += (~in_resp).sum()
        
        resp_id = chunk['response_tweet_id'].notnull()
        has_response_tweet_id += resp_id.sum()
        
        elapsed = time.time() - start_time
        print(f"  Chunk {i+1} processed: {total_rows:,} rows scanned ({elapsed:.1f}s elapsed)...")
        
    total_time = time.time() - start_time
    print(f"\nAudit complete in {total_time:.2f} seconds!")
    print(f"Total Rows: {total_rows:,}")
    
    print("\nColumn Null Analysis:")
    for col in col_names:
        n_cnt = null_counts[col]
        print(f"  - {col:25s}: {n_cnt:,} nulls ({n_cnt/total_rows*100:.2f}%)")
        
    print("\nInbound vs Outbound:")
    print(f"  - Inbound (Customer tweets): {inbound_counts[True]:,} ({inbound_counts[True]/total_rows*100:.2f}%)")
    print(f"  - Outbound (Brand tweets):   {inbound_counts[False]:,} ({inbound_counts[False]/total_rows*100:.2f}%)")
    
    print("\nConversation Linkages:")
    print(f"  - Root Tweets (in_response_to is null): {is_root_tweet:,} ({is_root_tweet/total_rows*100:.2f}%)")
    print(f"  - Reply Tweets (has in_response_to):    {has_in_response_to:,} ({has_in_response_to/total_rows*100:.2f}%)")
    print(f"  - Tweets that received responses:       {has_response_tweet_id:,} ({has_response_tweet_id/total_rows*100:.2f}%)")
    
    print("\nTop 20 Brands by Outbound Volume:")
    top_20 = brand_counts.most_common(20)
    for rank, (brand, count) in enumerate(top_20, 1):
        print(f"  {rank:2d}. {brand:20s}: {count:,} tweets")
        
    print("\nUber-Specific Findings:")
    uber_brands = {b: c for b, c in brand_counts.items() if 'uber' in str(b).lower()}
    print(f"  - Outbound authors matching 'uber': {uber_brands}")
    print(f"  - Total tweets mentioning 'uber' in text: {uber_text_mentions:,}")

    # Write summary report to json
    results = {
        "file_size_mb": round(file_size_mb, 2),
        "total_rows": int(total_rows),
        "columns": col_names,
        "null_counts": {k: int(v) for k, v in null_counts.items()},
        "inbound_counts": {str(k): int(v) for k, v in inbound_counts.items()},
        "root_tweets": int(is_root_tweet),
        "reply_tweets": int(has_in_response_to),
        "tweets_with_responses": int(has_response_tweet_id),
        "top_20_brands": {k: int(v) for k, v in top_20},
        "uber_brands": {k: int(v) for k, v in uber_brands.items()},
        "uber_text_mentions": int(uber_text_mentions)
    }
    
    os.makedirs("data/audit", exist_ok=True)
    with open("data/audit/audit_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved summary to data/audit/audit_summary.json")

if __name__ == "__main__":
    audit_sample("sample.csv")
    if os.path.exists("twcs.csv"):
        audit_full_dataset("twcs.csv")
    else:
        print("twcs.csv not found!")

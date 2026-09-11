"""
Brand Comparison Script
Compares the top customer support brands across key metrics:
1. Volume (Inbound & Outbound)
2. DM Deflection Rate (% of brand tweets telling customer to DM)
3. Substantive Resolution Content (URLs with help guides, specific instructions)
4. Multi-turn Depth (Customer replies back to brand)
5. Domain clarity (Ride-hailing vs Tech vs Retail)
"""

import re
import json
import pandas as pd
from collections import defaultdict

TARGET_BRANDS = ['Uber_Support', 'SpotifyCares', 'AppleSupport', 'AmazonHelp', 'Delta']

def compare_brands(csv_path="twcs.csv", chunk_size=250_000):
    print("Streaming twcs.csv to compare top brands...")
    
    brand_stats = {
        b: {
            "outbound_count": 0,
            "total_outbound_len": 0,
            "dm_deflection_count": 0,
            "url_link_count": 0,
            "has_response_count": 0, # brand tweet received customer follow-up
            "sample_responses": []
        }
        for b in TARGET_BRANDS
    }
    
    dm_regex = re.compile(r'\b(dm|direct message|private message)\b', re.IGNORECASE)
    url_regex = re.compile(r'https?://\S+')
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        # Filter for outbound tweets from target brands
        outbound = chunk[(chunk['inbound'] == False) & (chunk['author_id'].isin(TARGET_BRANDS))]
        
        for _, row in outbound.iterrows():
            b = row['author_id']
            text = str(row['text'])
            
            stats = brand_stats[b]
            stats["outbound_count"] += 1
            stats["total_outbound_len"] += len(text)
            
            if dm_regex.search(text):
                stats["dm_deflection_count"] += 1
                
            if url_regex.search(text):
                stats["url_link_count"] += 1
                
            if pd.notnull(row['response_tweet_id']):
                stats["has_response_count"] += 1
                
            if len(stats["sample_responses"]) < 3 and not dm_regex.search(text):
                stats["sample_responses"].append(text)
                
    # Calculate percentages and averages
    summary = {}
    for b, s in brand_stats.items():
        total = s["outbound_count"]
        if total == 0:
            continue
        summary[b] = {
            "total_outbound_tweets": total,
            "avg_tweet_char_length": round(s["total_outbound_len"] / total, 1),
            "dm_deflection_rate_pct": round(s["dm_deflection_count"] / total * 100, 2),
            "help_url_rate_pct": round(s["url_link_count"] / total * 100, 2),
            "follow_up_rate_pct": round(s["has_response_count"] / total * 100, 2),
            "non_dm_sample_responses": s["sample_responses"][:2]
        }
        
    print("\nBrand Comparison Summary:")
    print(json.dumps(summary, indent=2))
    
    with open("data/audit/brand_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    compare_brands()

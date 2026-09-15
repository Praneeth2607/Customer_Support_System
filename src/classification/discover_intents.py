"""
Step 4: Intent Discovery & Unsupervised Clustering

Analyzes customer queries from data/processed/uber_conversations.json:
1. Performs TF-IDF vectorization (unigrams and bigrams).
2. Extracts global top n-grams to identify dominant customer vocabulary.
3. Fits K-Means clustering (k=10) to discover natural problem groupings.
4. Extracts top keyword terms and real sample queries per cluster.
5. Exports discovery findings to data/audit/intent_discovery_report.json.

Re-run history:
  v1 (k=6, 20k random sample): produced clusters dominated by noise (47%) and
    an unlabeled Uber Eats topic (11.7%), with no cluster distinctly matching
    driver_conduct_and_safety or pickup_and_route_issue. Those two intents are
    real but low-frequency relative to fare/cancellation/account chatter, so
    unsupervised KMeans folds them into the largest generic cluster rather
    than isolating them -- expected behavior on imbalanced support text, not
    a bug, but it meant the "empirically derived" framing overstated what the
    clustering alone could show.
  v2 (k=10, full 42,186-query pool -- current): clustering on the full pool
    instead of a 20k subsample, and raising k from 6 to 10, cleanly separates
    fare/cancellation, lost-item, account/security (disabled/hacked), a
    pickup-wait cluster ("minutes away", "waiting"), and Uber Eats content
    from the residual generic/noise mass. This directly motivated adding the
    `out_of_scope_or_unclear` 7th taxonomy label (see docs/intent_taxonomy.md)
    to explicitly own the Eats + low-signal traffic instead of forcing it
    into one of the six ride-support intents.
"""

import os
import json
import time
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def discover_intents(
    input_json="data/processed/uber_conversations.json",
    output_report="data/audit/intent_discovery_report.json",
    num_clusters=10,
    sample_size=None,
    random_state=42
):
    print("=" * 70)
    print("STEP 4: EMPIRICAL INTENT DISCOVERY & CLUSTER ANALYSIS")
    print("=" * 70)
    
    start_time = time.time()
    with open(input_json, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations):,} conversations in {time.time() - start_time:.2f}s")
    
    # Extract clean queries
    queries = [c["first_customer_query_clean"] for c in conversations if len(c["first_customer_query_clean"].strip()) > 5]
    print(f"Extracted {len(queries):,} substantive customer queries.")
    
    # Sample only if a cap is explicitly requested; default is the full pool
    # (42,186 queries clusters in well under a minute, so no need to subsample).
    if sample_size is not None and len(queries) > sample_size:
        np.random.seed(random_state)
        sampled_indices = np.random.choice(len(queries), size=sample_size, replace=False)
        analysis_queries = [queries[i] for i in sampled_indices]
    else:
        analysis_queries = queries
        
    print(f"Analyzing sample of {len(analysis_queries):,} queries...")
    
    # 1. TF-IDF Vectorization
    print("Computing TF-IDF matrices (unigrams + bigrams)...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5,
        max_df=0.6
    )
    X = tfidf.fit_transform(analysis_queries)
    terms = np.array(tfidf.get_feature_names_out())
    
    # 2. Global Top N-Grams
    mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
    top_global_indices = mean_tfidf.argsort()[::-1][:25]
    top_global_terms = [
        {"term": str(terms[idx]), "score": round(float(mean_tfidf[idx]), 4)}
        for idx in top_global_indices
    ]
    
    # 3. K-Means Clustering
    print(f"Running K-Means clustering (k={num_clusters})...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X)
    
    # Analyze Clusters
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    cluster_counts = Counter(labels)
    
    clusters_data = []
    for i in range(num_clusters):
        top_terms = [str(terms[idx]) for idx in order_centroids[i, :12]]
        
        # Find queries closest to centroid
        cluster_indices = np.where(labels == i)[0]
        cluster_size = len(cluster_indices)
        cluster_pct = round(cluster_size / len(analysis_queries) * 100, 2)
        
        # Calculate distances to centroid
        centroid = kmeans.cluster_centers_[i]
        cluster_vectors = X[cluster_indices]
        # Euclidean distance
        distances = np.linalg.norm(cluster_vectors.toarray() - centroid, axis=1)
        closest_indices = cluster_indices[distances.argsort()[:5]]
        sample_queries = [analysis_queries[idx] for idx in closest_indices]
        
        cluster_info = {
            "cluster_id": i + 1,
            "size": cluster_size,
            "percentage": cluster_pct,
            "top_keywords": top_terms,
            "representative_queries": sample_queries
        }
        clusters_data.append(cluster_info)
        
    # Sort clusters by size
    clusters_data.sort(key=lambda c: c["size"], reverse=True)
    
    report = {
        "total_analyzed_queries": len(analysis_queries),
        "num_clusters": num_clusters,
        "top_global_ngrams": top_global_terms,
        "clusters": clusters_data
    }
    
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved Intent Discovery Report to {output_report}")
    
    print("\n" + "=" * 70)
    print("DISCOVERED CLUSTERS SUMMARY:")
    print("=" * 70)
    for c in clusters_data:
        print(f"\nCluster {c['cluster_id']} ({c['size']:,} queries, {c['percentage']}%):")
        print(f"  Keywords: {', '.join(c['top_keywords'][:8])}")
        sample_safe = c['representative_queries'][0][:100].encode('ascii', 'replace').decode('ascii')
        print(f"  Sample Query: {sample_safe}...")
        
    return report

if __name__ == "__main__":
    discover_intents()

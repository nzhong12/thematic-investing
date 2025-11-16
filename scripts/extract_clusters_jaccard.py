"""
Jaccard Distance + Hierarchical Clustering (Section 2.2 - Advanced Approach)
============================================================================

METHOD: Jaccard distance between correlation neighborhoods with hierarchical clustering
----------------------------------------------------------------------------------------
This script uses JACCARD METRIC between correlation neighborhoods:
- Measures similarity of correlation patterns (not just pairwise correlations)
- Uses HIERARCHICAL CLUSTERING (agglomerative, mid-level cut)
- More robust to noise, captures higher-order relationships

DIFFERENCE FROM extract_clusters.py:
------------------------------------
extract_clusters.py:  Uses ONLY pairwise correlations + basic threshold
                     Simple: "If corr(A,B) ≥ δ → connect A–B"

THIS FILE:           Uses JACCARD DISTANCE + HIERARCHICAL CLUSTERING
                     Advanced: Compares correlation neighborhoods, then clusters

METHOD (from paper Section 2.2):
---------------------------------
1. For each stock A: G_A = {X | r_AX > δ}  (A's correlation neighborhood)
2. Compute Jaccard distance: d(A,B) = 1 - |G_A ∩ G_B| / |G_A ∪ G_B|
3. Apply hierarchical clustering (agglomerative):
   - Start from singletons
   - Successively merge elements that are nearest according to d
   - Use average linkage method
4. Cut dendrogram at mid-level to determine final number of clusters
5. Merge singleton clusters into one 'outliers' group
   (elements whose returns are not significantly correlated to any other)

OUTPUTS (in scripts/outputs/):
-------------------------------
- jaccard_clusters_10day_2022-2024.txt  (daily clusters, one line per date)
- jaccard_clusters_30day_2022-2024.txt
- jaccard_clusters_50day_2022-2024.txt
- jaccard_cluster_summary_10day.txt     (statistics and examples)
- jaccard_cluster_summary_30day.txt
- jaccard_cluster_summary_50day.txt
- jaccard_clusters.pkl                  (Python dict of all clusters)
"""

import numpy as np
import pandas as pd
import pickle
import os
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ============================================================
# Configuration
# ============================================================
CORR_THRESHOLD = 0.6      # For defining neighborhoods: G_A = {X | r_AX > 0.6}
                          # 0.6 = high correlation for tight, meaningful neighborhoods
INPUT_FILE = 'correlation_data.pkl'
OUTPUT_DIR = 'outputs'

# ============================================================
# OPTIMIZATION OPTIONS - Adjust for faster testing
# ============================================================
SKIP_10DAY = True          # Skip 10-day window (faster, less useful anyway)
DATE_SAMPLING = 1          # Process every Nth date (1=all, 10=every 10th for fast testing)
                           # Jaccard is SLOW with 100 stocks - use DATE_SAMPLING=10 for testing

print("="*80)
print("JACCARD DISTANCE-BASED CLUSTERING (HIERARCHICAL)")
print("="*80)
print(f"\nConfiguration:")
print(f"  • Correlation threshold (neighborhood): {CORR_THRESHOLD}")
print(f"  • Clustering method: Hierarchical (average linkage, mid-level cut)")
print(f"  • Input: {INPUT_FILE}")
print(f"  • Output: {OUTPUT_DIR}/")

# ============================================================
# Load correlation data
# ============================================================
print(f"\nLoading correlation data from {INPUT_FILE}...")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found!")
    print(f"Please run: python scripts/sp500_rolling_correlation.py first")
    exit(1)

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)
    rolling_corrs = data['rolling_corrs']
    tickers = data['tickers']

print(f"✓ Loaded correlation data")
print(f"  • Tickers: {len(tickers)} stocks")
print(f"  • Windows: {sorted(rolling_corrs.keys())} days")

for window in sorted(rolling_corrs.keys()):
    dates = rolling_corrs[window].index.get_level_values(0).unique()
    print(f"  • {window}-day window: {len(dates)} dates ({dates[0]} to {dates[-1]})")

# ============================================================
# Helper Functions
# ============================================================

def compute_neighborhood(corr_matrix, stock, threshold):
    """
    Compute G_A = {X | r_AX > threshold} - the correlation neighborhood of stock A.
    
    Returns: set of tickers that have correlation > threshold with 'stock'
    """
    if stock not in corr_matrix.index:
        return set()
    
    stock_corrs = corr_matrix.loc[stock]
    neighbors = set(stock_corrs[stock_corrs > threshold].index) - {stock}
    return neighbors


def jaccard_distance(set_a, set_b):
    """
    Jaccard distance: d(A,B) = 1 - |G_A ∩ G_B| / |G_A ∪ G_B|
    
    Returns: float in [0, 1]
        0 = identical neighborhoods
        1 = completely different neighborhoods
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 1.0
    
    return 1.0 - (intersection / union)


def build_jaccard_distance_matrix(corr_matrix, corr_threshold):
    """
    Build NxN Jaccard distance matrix.
    
    For each pair (A, B):
    1. G_A = {X | r_AX > corr_threshold}
    2. G_B = {X | r_BX > corr_threshold}
    3. d(A,B) = 1 - |G_A ∩ G_B| / |G_A ∪ G_B|
    """
    stocks = corr_matrix.index.tolist()
    n = len(stocks)
    
    # Compute all neighborhoods first
    neighborhoods = {stock: compute_neighborhood(corr_matrix, stock, corr_threshold) 
                     for stock in stocks}
    
    # Build distance matrix
    dist_matrix = pd.DataFrame(0.0, index=stocks, columns=stocks)
    
    for i, stock_a in enumerate(stocks):
        for j, stock_b in enumerate(stocks):
            if i < j:
                dist = jaccard_distance(neighborhoods[stock_a], neighborhoods[stock_b])
                dist_matrix.loc[stock_a, stock_b] = dist
                dist_matrix.loc[stock_b, stock_a] = dist
    
    return dist_matrix


def hierarchical_clustering_jaccard(jaccard_dist_matrix):
    """
    Apply hierarchical clustering using Jaccard distances.
    
    Following paper Section 2.2:
    1. Agglomerative hierarchical clustering (builds from singletons)
    2. Cut dendrogram at mid-level (average method)
    3. Merge singleton clusters into 'outliers' group
    
    Returns: list of clusters, where each cluster is a sorted list of tickers
             Last cluster labeled as 'outliers' if it contains singletons
    """
    stocks = jaccard_dist_matrix.index.tolist()
    n = len(stocks)
    
    if n < 2:
        return [stocks]
    
    # Convert distance matrix to condensed form for scipy
    # Extract upper triangle (excluding diagonal)
    condensed_dist = squareform(jaccard_dist_matrix.values, checks=False)
    
    # Hierarchical clustering using average linkage (most common for this application)
    Z = linkage(condensed_dist, method='average')
    
    # Cut dendrogram at mid-level
    # Use distance threshold = half of max distance in linkage matrix
    max_dist = Z[:, 2].max()
    mid_level_dist = max_dist / 2.0
    
    # Get cluster labels
    labels = fcluster(Z, t=mid_level_dist, criterion='distance')
    
    # Group stocks by cluster label
    cluster_dict = defaultdict(list)
    for stock, label in zip(stocks, labels):
        cluster_dict[label].append(stock)
    
    # Convert to list of clusters
    clusters = [sorted(stocks) for stocks in cluster_dict.values()]
    
    # Identify singletons and merge into outliers group
    singletons = []
    multi_stock_clusters = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            singletons.extend(cluster)
        else:
            multi_stock_clusters.append(cluster)
    
    # Sort multi-stock clusters by size (largest first)
    multi_stock_clusters.sort(key=len, reverse=True)
    
    # Add outliers group at the end if any singletons exist
    if singletons:
        multi_stock_clusters.append(sorted(singletons))
    
    return multi_stock_clusters

# ============================================================
# Main Clustering Loop
# ============================================================
print("\n" + "="*80)
print("CLUSTERING ANALYSIS")
print("="*80)

all_clusters = {}
cluster_stats = defaultdict(list)

# Apply optimization: skip 10-day window if enabled
windows_to_process = [w for w in sorted(rolling_corrs.keys()) if not (SKIP_10DAY and w == 10)]
if SKIP_10DAY and 10 in rolling_corrs.keys():
    print("\n⚡ OPTIMIZATION: Skipping 10-day window (set SKIP_10DAY=False to include)")
if DATE_SAMPLING > 1:
    print(f"⚡ OPTIMIZATION: Sampling every {DATE_SAMPLING}th date (set DATE_SAMPLING=1 for all dates)")

for window in windows_to_process:
    print(f"\n{'='*80}")
    print(f"Processing {window}-day Rolling Window")
    print(f"{'='*80}")
    
    corr_df = rolling_corrs[window]
    dates = sorted(corr_df.index.get_level_values(0).unique())
    
    # Apply date sampling if enabled
    if DATE_SAMPLING > 1:
        dates = dates[::DATE_SAMPLING]
        print(f"⚡ Using sampled dates: {len(dates)} dates (every {DATE_SAMPLING}th)")
    
    print(f"Analyzing {len(dates)} dates...")
    
    for i, date in enumerate(dates):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dates)} dates...")
        
        corr_matrix = corr_df.loc[date]
        
        # Build Jaccard distance matrix
        jaccard_dist_matrix = build_jaccard_distance_matrix(corr_matrix, CORR_THRESHOLD)
        
        # Apply hierarchical clustering
        clusters = hierarchical_clustering_jaccard(jaccard_dist_matrix)
        
        all_clusters[(window, date)] = clusters
        cluster_stats[window].append(len(clusters))
    
    print(f"✓ Completed {window}-day window")
    print(f"  • Avg clusters per date: {np.mean(cluster_stats[window]):.1f}")
    print(f"  • Min/Max clusters: {min(cluster_stats[window])}/{max(cluster_stats[window])}")

# ============================================================
# Save Results
# ============================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save pickle
output_file = os.path.join(OUTPUT_DIR, 'jaccard_clusters.pkl')
print(f"\nSaving clusters to {output_file}...")
with open(output_file, 'wb') as f:
    pickle.dump({
        'clusters': all_clusters,
        'tickers': tickers,
        'corr_threshold': CORR_THRESHOLD,
        'clustering_method': 'hierarchical_average_linkage',
        'windows': sorted(rolling_corrs.keys())
    }, f)
print(f"✓ Saved {len(all_clusters)} cluster snapshots to pickle")

# Export daily clusters to TXT files
print("\nExporting clusters to TXT files...")
for window in windows_to_process:
    txt_file = os.path.join(OUTPUT_DIR, f'jaccard_clusters_{window}day_2022-2024.txt')
    
    dates = sorted(rolling_corrs[window].index.get_level_values(0).unique())
    
    # Calculate summary statistics
    all_cluster_counts = [len(all_clusters[(window, d)]) for d in dates]
    all_cluster_sizes = []
    for d in dates:
        for cluster in all_clusters[(window, d)]:
            all_cluster_sizes.append(len(cluster))
    
    # Calculate cluster composition frequencies
    from collections import Counter
    cluster_compositions = []
    for d in dates:
        for cluster in all_clusters[(window, d)]:
            cluster_tuple = tuple(sorted(cluster))  # Sort for consistent comparison
            cluster_compositions.append(cluster_tuple)
    
    composition_counts = Counter(cluster_compositions)
    total_days = len(dates)
    
    with open(txt_file, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write(f"DAILY JACCARD-BASED CLUSTERS - {window}-day Rolling Window\n")
        f.write("="*80 + "\n\n")
        f.write(f"Period: 2022-01-01 to 2024-12-31\n")
        f.write(f"Total trading days: {len(dates)}\n")
        f.write(f"Correlation threshold (for neighborhoods): {CORR_THRESHOLD}\n")
        f.write(f"Clustering: Hierarchical (average linkage, mid-level cut)\n")
        f.write(f"Note: Singletons merged into outliers group\n\n")
        f.write(f"SUMMARY STATISTICS:\n")
        f.write(f"  • Avg clusters per day: {np.mean(all_cluster_counts):.2f}\n")
        f.write(f"  • Std dev: {np.std(all_cluster_counts):.2f}\n")
        f.write(f"  • Min/Max clusters: {min(all_cluster_counts)}/{max(all_cluster_counts)}\n")
        if all_cluster_sizes:
            f.write(f"  • Avg cluster size: {np.mean(all_cluster_sizes):.2f} stocks\n")
            f.write(f"  • Largest cluster ever: {max(all_cluster_sizes)} stocks\n")
        
        # Most frequent cluster compositions (only show multi-stock clusters)
        f.write(f"\n" + "="*80 + "\n")
        f.write(f"MOST FREQUENT MULTI-STOCK CLUSTER COMPOSITIONS\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Top 20 cluster groups (size ≥ 2) that appear most often across all {total_days} trading days:\n")
        f.write(f"(Singletons excluded - they represent stocks not clustering with others)\n\n")
        
        # Filter to only show clusters with 2+ stocks
        multi_stock_clusters = [(cluster, freq) for cluster, freq in composition_counts.most_common() 
                                 if len(cluster) >= 2]
        
        if len(multi_stock_clusters) == 0:
            f.write("No multi-stock clusters found. All stocks appear as singletons.\n")
            f.write("Consider lowering CORR_THRESHOLD to form more clusters.\n")
        else:
            for i, (cluster_tuple, freq) in enumerate(multi_stock_clusters[:20], 1):
                cluster_str = ', '.join(cluster_tuple)
                cluster_size = len(cluster_tuple)
                percentage = (freq / total_days) * 100
                f.write(f"{i}. ({cluster_str}) [size={cluster_size}]: {freq}/{total_days} days ({percentage:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DAILY CLUSTERS (one line per trading day)\n")
        f.write("="*80 + "\n\n")
        f.write("Format: YYYY-MM-DD [N clusters] | cluster1 | cluster2 | cluster3 | ...\n")
        f.write("        (within each cluster, stocks are comma-separated)\n\n")
        
        # Daily data
        for date in dates:
            clusters = all_clusters[(window, date)]
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            num_clusters = len(clusters)
            
            if len(clusters) > 0:
                cluster_str = ' | '.join([','.join(cluster) for cluster in clusters])
            else:
                cluster_str = '(no clusters found)'
            
            f.write(f"{date_str} [{num_clusters} clusters] | {cluster_str}\n")
    
    print(f"✓ Saved {len(dates)} days of clusters to: {txt_file}")

# Generate summary reports
print("\nGenerating summary reports...")

for window in windows_to_process:
    output_file = os.path.join(OUTPUT_DIR, f'jaccard_cluster_summary_{window}day.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"JACCARD CLUSTER ANALYSIS SUMMARY - {window}-day Rolling Window\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis Period: 2022-01-01 to 2024-12-31\n")
        f.write(f"Correlation Threshold (neighborhoods): {CORR_THRESHOLD}\n")
        f.write(f"Method: Hierarchical Clustering (average linkage, mid-level cut)\n")
        f.write(f"        - Jaccard distance between correlation neighborhoods\n")
        f.write(f"        - Singletons merged into outliers group\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERVIEW\n")
        f.write("="*80 + "\n\n")
        
        window_clusters = [len(all_clusters[(window, d)]) for d in 
                          rolling_corrs[window].index.get_level_values(0).unique()]
        
        f.write(f"Number of dates analyzed: {len(window_clusters)}\n")
        f.write(f"Average clusters per date: {np.mean(window_clusters):.2f}\n")
        f.write(f"Std deviation: {np.std(window_clusters):.2f}\n")
        f.write(f"Min/Max clusters: {min(window_clusters)} / {max(window_clusters)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("EXAMPLE CLUSTERS (Most Recent Date)\n")
        f.write("="*80 + "\n\n")
        
        dates = sorted(rolling_corrs[window].index.get_level_values(0).unique())
        latest_date = dates[-1]
        latest_clusters = all_clusters[(window, latest_date)]
        
        date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        f.write(f"Date: {date_str}\n")
        f.write(f"Total clusters found: {len(latest_clusters)}\n\n")
        
        if len(latest_clusters) > 0:
            f.write(f"Largest clusters:\n")
            for i, cluster in enumerate(latest_clusters[:10], 1):
                f.write(f"\nCluster {i} ({len(cluster)} stocks):\n")
                f.write(f"  {', '.join(cluster)}\n")
        else:
            f.write("No clusters found (all Jaccard distances above threshold).\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Clusters are formed based on Jaccard distance between correlation neighborhoods.\n")
        f.write(f"Two stocks are in same cluster if they have similar correlation patterns\n")
        f.write(f"(i.e., their neighborhoods G_A and G_B have high overlap).\n\n")
        f.write("This method is more robust than pairwise correlations because:\n")
        f.write("  • Captures higher-order relationships\n")
        f.write("  • Less sensitive to individual correlation noise\n")
        f.write("  • Groups stocks with similar correlation structures\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Saved summary to {output_file}")

# ============================================================
# Print Final Summary
# ============================================================
print("\n" + "="*80)
print("CLUSTERING COMPLETE")
print("="*80)

print(f"\nTotal cluster snapshots: {len(all_clusters)}")
print(f"\nCluster statistics by window:")
for window in windows_to_process:
    avg_clusters = np.mean(cluster_stats[window])
    std_clusters = np.std(cluster_stats[window])
    print(f"  • {window}-day: {avg_clusters:.1f} ± {std_clusters:.1f} clusters per date")

print(f"\nOutputs saved to:")
print(f"  • {os.path.join(OUTPUT_DIR, 'jaccard_clusters.pkl')}")
for window in windows_to_process:
    print(f"  • {os.path.join(OUTPUT_DIR, f'jaccard_clusters_{window}day_2022-2024.txt')}")
    print(f"  • {os.path.join(OUTPUT_DIR, f'jaccard_cluster_summary_{window}day.txt')}")

print("\n✓ Done!")
print("\nNext steps:")
print("  • Compare with correlation-based clusters (extract_clusters.py)")
print("  • Analyze cluster stability over time (2.3 of paper)")
print("  • More next steps in ReadMe and paper sections after 2.2")
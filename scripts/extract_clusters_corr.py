"""
Basic Correlation Threshold Clustering (Section 2.2 - Simple Approach)
======================================================================

METHOD: Basic correlation-based clustering with MST
----------------------------------------------------
This script uses PAIRWISE CORRELATIONS directly with a simple threshold:
- If corr(A,B) ≥ δ → connect A–B directly
- Uses MST-based clustering (threshold + connected components)

DIFFERENCE FROM extract_clusters_jaccard.py:
---------------------------------------------
THIS FILE:                Uses pairwise correlations + basic threshold
                         Simple and fast, but sensitive to noise

extract_clusters_jaccard.py:  Uses Jaccard distance + hierarchical clustering
                              More robust, captures higher-order relationships

This script:
1. Loads pre-computed correlation data from correlation_data.pkl
2. For each rolling window (10, 30, 50 days) and each date:
   - Builds MST using distance metric: d_ij = sqrt(2 * (1 - corr_ij))
   - Filters MST edges by correlation threshold (keeps only corr ≥ 0.6)
   - Identifies connected components as market clusters/themes
3. Saves clusters to outputs/corr_clusters.pkl
4. Exports human-readable cluster summaries to outputs/corr_cluster_summary_*.txt

This is efficient because MST has only (n-1) edges instead of O(n²) pairwise correlations.

HOW IT WORKS (3-step process):
-------------------------------
1. BUILD MST: For each date, convert 20×20 correlation matrix into MST (19 edges connecting 20 stocks)
   - Uses distance metric: d_ij = sqrt(2 * (1 - corr_ij))
   - MST captures the strongest correlation structure with minimal edges

2. FILTER EDGES: Remove weak MST edges (correlation < 0.6)
   - This breaks the MST into disconnected pieces
   - Only keeps edges between stocks that move strongly together

3. EXTRACT CLUSTERS: Find connected components in filtered graph
   - Each disconnected group = one cluster of highly correlated stocks
   - Example: If A—B—C are connected and D—E are connected (but separated), 
     that's 2 clusters: [A,B,C] and [D,E]
   - Isolated stocks (no strong edges) become singleton clusters

This runs ~2,250 times: 3 windows (10/30/50 days) × ~750 trading days = 2,250 cluster snapshots.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
import os
from collections import defaultdict
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
CORRELATION_THRESHOLD = 0.52  # Minimum correlation to keep in cluster (δ in paper)
                              # 0.52 = moderate-high correlation for balanced cluster sizes
INPUT_FILE = 'correlation_data.pkl'
OUTPUT_DIR = 'outputs'  # Relative to scripts/ folder where this file is located
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'corr_clusters.pkl')

# ============================================================
# OPTIMIZATION OPTIONS - Adjust for faster testing
# ============================================================
SKIP_10DAY = True          # Skip 10-day window (faster, less useful anyway)
DATE_SAMPLING = 1          # Process every Nth date (1=all, 10=every 10th, etc.)
                           # Set to 10 for fast testing, 1 for production

print("="*80)
print("TIME SERIES CLUSTERING USING MST")
print("="*80)
print(f"\nConfiguration:")
print(f"  • Correlation threshold (δ): {CORRELATION_THRESHOLD}")
print(f"  • Input: {INPUT_FILE}")
print(f"  • Output: {OUTPUT_FILE}")

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

# Get date ranges for each window
for window in sorted(rolling_corrs.keys()):
    dates = rolling_corrs[window].index.get_level_values(0).unique()
    print(f"  • {window}-day window: {len(dates)} dates ({dates[0]} to {dates[-1]})")

# ============================================================
# Helper Functions
# ============================================================

def build_mst(corr_matrix):
    """
    Build Minimum Spanning Tree from a single correlation matrix snapshot.
    
    IMPORTANT: This function builds ONE MST for ONE specific date/window combination.
    The main loop calls this ~2,250 times total:
      - 3 windows (10, 30, 50 days) × ~750 trading days (2022-2024) = 2,250 MSTs
    
    Each MST represents the correlation structure for that specific:
      - Rolling window size (10/30/50 days of returns)
      - Trading date (one snapshot in time)
    
    Distance metric: d_ij = sqrt(2 * (1 - corr_ij))
      Maps: corr=+1 → d=0, corr=0 → d=√2, corr=-1 → d=2
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        NxN correlation matrix for one date (N = number of stocks)
    
    Returns:
    --------
    mst : networkx.Graph
        Minimum spanning tree with (N-1) edges, weights = distances
    """
    corr_matrix = corr_matrix.fillna(0)
    
    # Convert correlation to distance
    dist_matrix = np.sqrt(2 * (1 - corr_matrix.clip(-1, 1)))
    dist_matrix = dist_matrix.fillna(2.0)
    np.fill_diagonal(dist_matrix.values, 0.0)
    
    # Build MST using Kruskal's algorithm
    G = nx.from_pandas_adjacency(dist_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    mst = nx.minimum_spanning_tree(G)
    
    return mst

def filter_mst_by_correlation(mst, threshold):
    """
    Keep only MST edges with correlation ≥ threshold.
    
    This implements the thresholding step: r_ij > δ
    
    Parameters:
    -----------
    mst : networkx.Graph
        Minimum spanning tree with edge weights as distances
    threshold : float
        Minimum correlation to keep (δ in paper, typically 0.6)
    
    Returns:
    --------
    filtered_mst : networkx.Graph
        MST with only high-correlation edges
    """
    filtered_mst = nx.Graph()
    filtered_mst.add_nodes_from(mst.nodes())
    
    for u, v, data in mst.edges(data=True):
        distance = data['weight']
        # Convert distance back to correlation: corr = 1 - (d²/2)
        correlation = 1 - (distance**2) / 2
        
        # Keep only edges with strong positive correlation
        if correlation >= threshold:
            filtered_mst.add_edge(u, v, weight=distance, correlation=correlation)
    
    return filtered_mst

def extract_clusters(filtered_mst):
    """
    Extract connected components as clusters.
    
    After filtering MST by correlation threshold, connected components
    represent groups of stocks that move together (market themes).
    
    This approximates the paper's cluster definition:
        G_A = { X | r_AX > δ }
    
    Parameters:
    -----------
    filtered_mst : networkx.Graph
        MST filtered to only include edges with corr ≥ threshold
    
    Returns:
    --------
    clusters : list of lists
        Each sublist contains tickers in one cluster
        Sorted by cluster size (largest first)
    """
    # Find connected components
    components = list(nx.connected_components(filtered_mst))
    
    # Convert to sorted lists (largest clusters first)
    clusters = [sorted(list(component)) for component in components]
    clusters.sort(key=len, reverse=True)
    
    return clusters

# ============================================================
# Main Clustering Loop
# ============================================================
print("\n" + "="*80)
print("CLUSTERING ANALYSIS")
print("="*80)

# Dictionary to store all clusters: clusters[(window, date)] = [[cluster1], [cluster2], ...]
all_clusters = {}

# Statistics tracking
cluster_stats = defaultdict(list)  # cluster_stats[window] = [num_clusters_per_date]

for window in sorted(rolling_corrs.keys()):
    print(f"\n{'='*80}")
    print(f"Processing {window}-day Rolling Window")
    print(f"{'='*80}")
    
    corr_df = rolling_corrs[window]
    dates = sorted(corr_df.index.get_level_values(0).unique())
    
    # Apply date sampling if enabled
    if DATE_SAMPLING > 1:
        dates = dates[::DATE_SAMPLING]
        print(f"⚡ OPTIMIZATION: Sampling every {DATE_SAMPLING}th date ({len(dates)} dates total)")
    
    print(f"Analyzing {len(dates)} dates...")
    
    for i, date in enumerate(dates):
        # Progress indicator (every 100 dates)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dates)} dates...")
        
        # Extract correlation matrix for this date
        corr_matrix = corr_df.loc[date]
        
        # Build MST
        mst = build_mst(corr_matrix)
        
        # Filter MST by correlation threshold
        filtered_mst = filter_mst_by_correlation(mst, CORRELATION_THRESHOLD)
        
        # Extract clusters (connected components)
        clusters = extract_clusters(filtered_mst)
        
        # Store clusters
        all_clusters[(window, date)] = clusters
        
        # Track statistics
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

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save clusters dictionary to pickle
print(f"\nSaving clusters to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({
        'clusters': all_clusters,
        'tickers': tickers,
        'threshold': CORRELATION_THRESHOLD,
        'windows': sorted(rolling_corrs.keys())
    }, f)
print(f"✓ Saved {len(all_clusters)} cluster snapshots to pickle")

# Export clusters to TXT files (one per window) with summary at top
print("\nExporting clusters to TXT files...")
for window in sorted(rolling_corrs.keys()):
    txt_file = os.path.join(OUTPUT_DIR, f'corr_clusters_{window}day_2022-2024.txt')
    
    # Get all dates for this window
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
    
    # Write to TXT file
    with open(txt_file, 'w') as f:
        # Header with summary
        f.write("="*80 + "\n")
        f.write(f"DAILY CLUSTERS - {window}-day Rolling Window\n")
        f.write("="*80 + "\n\n")
        f.write(f"Period: 2022-01-01 to 2024-12-31\n")
        f.write(f"Total trading days: {len(dates)}\n")
        f.write(f"Correlation threshold: {CORRELATION_THRESHOLD}\n\n")
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
            f.write("Consider lowering CORRELATION_THRESHOLD to form more clusters.\n")
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
        
        # One line per date with clusters
        for date in dates:
            clusters = all_clusters[(window, date)]
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            num_clusters = len(clusters)
            
            # Format clusters
            if len(clusters) > 0:
                cluster_str = ' | '.join([','.join(cluster) for cluster in clusters])
            else:
                cluster_str = '(no clusters found)'
            
            f.write(f"{date_str} [{num_clusters} clusters] | {cluster_str}\n")
    
    print(f"✓ Saved {len(dates)} days of clusters to: {txt_file}")
    print(f"  • Format: Summary at top, then ~750 lines (one per trading day)")
    print(f"  • Avg clusters per day: {np.mean(all_cluster_counts):.2f}")

# ============================================================
# Generate Summary Reports
# ============================================================
print("\nGenerating summary reports...")

for window in sorted(rolling_corrs.keys()):
    output_file = os.path.join(OUTPUT_DIR, f'corr_cluster_summary_{window}day.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CLUSTER ANALYSIS SUMMARY - {window}-day Rolling Window\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis Period: 2022-01-01 to 2024-12-31\n")
        f.write(f"Correlation Threshold (δ): {CORRELATION_THRESHOLD}\n")
        f.write(f"Method: MST + Threshold + Connected Components\n\n")
        
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
        
        # Get most recent date
        dates = sorted(rolling_corrs[window].index.get_level_values(0).unique())
        latest_date = dates[-1]
        latest_clusters = all_clusters[(window, latest_date)]
        
        date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        f.write(f"Date: {date_str}\n")
        f.write(f"Total clusters found: {len(latest_clusters)}\n\n")
        
        if len(latest_clusters) > 0:
            f.write(f"Largest clusters:\n")
            for i, cluster in enumerate(latest_clusters[:10], 1):  # Top 10 clusters
                f.write(f"\nCluster {i} ({len(cluster)} stocks):\n")
                f.write(f"  {', '.join(cluster)}\n")
        else:
            f.write("No clusters found (all correlations below threshold).\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Clusters represent groups of stocks with correlation ≥ {CORRELATION_THRESHOLD}\n")
        f.write("over the specified rolling window. These are likely:\n")
        f.write("  • Sector-based groupings (e.g., tech stocks, utilities)\n")
        f.write("  • Theme-based groupings (e.g., growth vs. value)\n")
        f.write("  • Market regime indicators (more/fewer clusters = fragmentation/cohesion)\n\n")
        f.write("Large clusters indicate broad market themes.\n")
        f.write("Many small clusters indicate market fragmentation.\n")
        f.write("Cluster stability over time indicates persistent themes.\n")
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
for window in sorted(rolling_corrs.keys()):
    avg_clusters = np.mean(cluster_stats[window])
    std_clusters = np.std(cluster_stats[window])
    print(f"  • {window}-day: {avg_clusters:.1f} ± {std_clusters:.1f} clusters per date")

print(f"\nOutputs saved to:")
print(f"  • {OUTPUT_FILE}")
for window in sorted(rolling_corrs.keys()):
    print(f"  • {os.path.join(OUTPUT_DIR, f'corr_cluster_summary_{window}day.txt')}")

print("\n✓ Done!")
print("\nNext steps:")
print("  • Examine cluster summaries in scripts/outputs/")
print("  • Analyze cluster stability over time")
print("  • Compare cluster composition across different windows")
print("  • Use clusters.pkl for downstream analysis")


"""
MST Visualization ONLY
======================
Loads pre-computed correlation data and shows MST visualization.
NO statistics printed.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os

# ============================================================
# Load pre-computed data
# ============================================================
print("Loading correlation data...")

# Try to load from pickle file if it exists
data_file = 'correlation_data.pkl'

if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        rolling_corrs = data['rolling_corrs']
        tickers = data['tickers']
    print(f"✓ Loaded from {data_file}")
else:
    # Import from session if available
    try:
        rolling_corrs
        tickers
        print("✓ Using data from current session")
    except NameError:
        print(f"ERROR: No data found!")
        print(f"Please run: python sp500_rolling_correlation.py first")
        exit(1)

# ============================================================
# Build MST
# ============================================================
window = 30
corr_df = rolling_corrs[window]
latest_date = corr_df.index.get_level_values(0)[-1]
corr_matrix = corr_df.loc[latest_date]

# Check for NaN values and fill them
if corr_matrix.isnull().any().any():
    print("⚠️  Found NaN values in correlation matrix, filling with 0...")
    corr_matrix = corr_matrix.fillna(0)

# Convert to distance and build MST
# Clip correlation values to [-1, 1] to avoid sqrt of negative numbers
corr_matrix_clipped = corr_matrix.clip(-1, 1)
dist_matrix = np.sqrt(2 * (1 - corr_matrix_clipped))

# Replace any remaining NaN with large distance
dist_matrix = dist_matrix.fillna(2.0)

G = nx.from_pandas_adjacency(dist_matrix)
mst = nx.minimum_spanning_tree(G)

print(f"✓ MST built: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges")

# ============================================================
# Visualize ONLY
# ============================================================
edges_sorted = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'])

fig, ax = plt.subplots(figsize=(14, 10))

# Position nodes
pos = nx.spring_layout(mst, k=2, iterations=50, seed=42)

# Draw nodes
nx.draw_networkx_nodes(mst, pos, 
                       node_color='lightblue',
                       node_size=1500,
                       alpha=0.9,
                       ax=ax)

# Draw edges with thickness proportional to 1/distance
edges = mst.edges()
weights = [mst[u][v]['weight'] for u, v in edges]
max_width = 6
min_width = 0.5
widths = [max_width / (w + 0.1) for w in weights]

nx.draw_networkx_edges(mst, pos,
                       width=widths,
                       alpha=0.6,
                       edge_color='gray',
                       ax=ax)

# Draw labels
nx.draw_networkx_labels(mst, pos,
                       font_size=10,
                       font_weight='bold',
                       ax=ax)

# Add edge labels for top 5 connections
edge_labels = {}
for node1, node2, data in edges_sorted[:5]:
    edge_labels[(node1, node2)] = f"{data['weight']:.2f}"

nx.draw_networkx_edge_labels(mst, pos,
                             edge_labels,
                             font_size=8,
                             font_color='red',
                             ax=ax)

date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
ax.set_title(f'Minimum Spanning Tree - {window}-day Rolling Correlation\n'
             f'Date: {date_str}\n'
             f'Edge thickness ∝ 1/distance (thicker = stronger correlation)',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
print("✓ Showing MST visualization...")
plt.show()

print("✓ Done")

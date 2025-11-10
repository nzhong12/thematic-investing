"""
MST Demo: Minimum Spanning Tree from Rolling Correlations
===========================================================
Takes the rolling correlation data from sp500_rolling_correlation.py output
and builds an MST to identify the strongest stock relationships.

USAGE: 
    First run: python sp500_rolling_correlation.py
    Then run: python mst_demo.py
    
    Or import rolling_corrs directly if running in same session.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

print("="*80)
print("MINIMUM SPANNING TREE (MST) FROM ROLLING CORRELATIONS")
print("="*80)

# ============================================================
# STEP 1: Import rolling correlation data from main script
# ============================================================
print("\n" + "="*80)
print("STEP 1: Loading rolling correlation data")
print("="*80)

# Add src directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import from sp500_rolling_correlation if it was run in same session
try:
    # If running in interactive session after sp500_rolling_correlation.py
    print("\nAttempting to use rolling_corrs from current session...")
    rolling_corrs
    tickers
    print(f"✓ Found rolling_corrs in current session")
    print(f"✓ Windows available: {list(rolling_corrs.keys())}")
    print(f"✓ Tickers: {tickers}")
except NameError:
    # If running as standalone script, need to compute correlations
    print("\n⚠️  rolling_corrs not found in session")
    print("Loading data and computing correlations...")
    
    import wrds
    
    start_date = "2024-01-01"
    end_date = "2025-01-20"
    num_stocks = 20
    
    print("\nConnecting to WRDS...")
    db = wrds.Connection()
    
    # Get S&P 500 stocks
    sp500_query = """
        SELECT DISTINCT a.permno, b.ticker
        FROM crsp.dsp500list as a
        LEFT JOIN crsp.dsenames as b
        ON a.permno = b.permno
            AND b.namedt <= '{end_date}'
            AND b.nameendt >= '{start_date}'
        WHERE a.ending >= '{start_date}'
            AND b.ticker IS NOT NULL
        ORDER BY a.permno
        LIMIT 100
    """.format(start_date=start_date, end_date=end_date)
    
    sp500_df = db.raw_sql(sp500_query)
    tickers = sp500_df['ticker'].dropna().unique()[:num_stocks].tolist()
    print(f"✓ Selected {len(tickers)} stocks: {tickers}")
    
    # Download price data
    query = """
        SELECT 
            a.date,
            a.permno,
            b.ticker,
            a.prc
        FROM 
            crsp.dsf as a
        LEFT JOIN 
            crsp.dsenames as b
        ON 
            a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
        WHERE 
            a.date BETWEEN '{start_date}' AND '{end_date}'
            AND b.ticker IN ({ticker_list})
            AND a.prc IS NOT NULL
        ORDER BY a.date, b.ticker
    """.format(
        start_date=start_date,
        end_date=end_date,
        ticker_list=','.join(f"'{t}'" for t in tickers)
    )
    
    print("Downloading data from WRDS...")
    df = db.raw_sql(query)
    db.close()
    
    print(f"✓ Downloaded {len(df)} rows")
    
    # Compute returns
    print("Computing daily returns...")
    prices = df.pivot(index='date', columns='ticker', values='prc').abs()
    returns = prices.pct_change().dropna()
    print(f"✓ Returns shape: {returns.shape}")
    
    # Compute rolling correlations for multiple windows
    print("\nComputing rolling correlations...")
    rolling_corrs = {}
    for window in [10, 30, 50]:
        rolling_corr = returns.rolling(window=window).corr()
        rolling_corrs[window] = rolling_corr
        print(f"  ✓ {window}-day: {len(rolling_corr.index.get_level_values(0).unique())} matrices")

# ============================================================
# STEP 2: Extract latest 30-day correlation matrix
# ============================================================
print("\n" + "="*80)
print("STEP 2: Building MST from 30-day correlations")
print("="*80)

# Use 30-day window for MST
window = 30
corr_df = rolling_corrs[window]
latest_date = corr_df.index.get_level_values(0)[-1]
corr_matrix = corr_df.loc[latest_date]

date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
print(f"\n✓ Using date: {date_str}")
print(f"✓ Stocks: {corr_matrix.shape[0]}")

# Convert to distance matrix
# Distance formula: dist_ij = sqrt(2 * (1 - corr_ij))
dist_matrix = np.sqrt(2 * (1 - corr_matrix))

# Create graph from distance matrix and compute MST
G = nx.from_pandas_adjacency(dist_matrix)
mst = nx.minimum_spanning_tree(G)

print(f"✓ MST computed: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges")

# Sort edges by weight for visualization
edges_sorted = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'])

# ============================================================
# STEP 3: Visualize MST
# ============================================================
print("\n" + "="*80)
print("STEP 3: Visualizing MST")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 10))

# Position nodes using spring layout
pos = nx.spring_layout(mst, k=2, iterations=50, seed=42)

# Draw nodes
nx.draw_networkx_nodes(mst, pos, 
                       node_color='lightblue',
                       node_size=1500,
                       alpha=0.9,
                       ax=ax)

# Draw edges with thickness proportional to 1/distance (stronger = thicker)
edges = mst.edges()
weights = [mst[u][v]['weight'] for u, v in edges]

# Convert distance to edge width: smaller distance = thicker edge
# width = k / distance (where k is a scaling factor)
max_width = 6
min_width = 0.5
widths = [max_width / (w + 0.1) for w in weights]  # +0.1 to avoid division by zero

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

# Add edge labels showing distances for strongest connections
edge_labels = {}
for node1, node2, data in edges_sorted[:5]:
    edge_labels[(node1, node2)] = f"{data['weight']:.2f}"

nx.draw_networkx_edge_labels(mst, pos,
                             edge_labels,
                             font_size=8,
                             font_color='red',
                             ax=ax)

ax.set_title(f'Minimum Spanning Tree - {window}-day Rolling Correlation\n'
             f'Date: {date_str}\n'
             f'Edge thickness ∝ 1/distance (thicker = stronger correlation)',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()

print("\n✓ Visualization created")
print("  - Node size represents stocks")
print("  - Edge thickness ∝ 1/distance (thicker = more correlated)")
print("  - Red labels show distances for top 5 strongest connections")
print("\nShowing plot...")

plt.show()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ MST Complete - showing visualization")
print("="*80)

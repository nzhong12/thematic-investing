"""
MST Visualization ONLY
======================
Loads pre-computed correlation data and shows MST visualization.
NO statistics printed.

NAVIGATION:
- Left/Right arrows: Switch between 10/30/50-day windows
- Up/Down arrows: Navigate through recent dates
- Q or Escape: Quit
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('macosx')  # Use macOS native backend for arrow key navigation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
# Setup navigation state
# ============================================================
windows = sorted(rolling_corrs.keys())
current_window_idx = windows.index(30) if 30 in windows else 0

# Get available dates (ALL dates - no limit, can navigate through entire dataset)
all_dates = rolling_corrs[windows[current_window_idx]].index.get_level_values(0).unique()
available_dates = sorted(all_dates)  # All dates from 2022-2024 (~750 trading days)
current_date_idx = len(available_dates) - 1  # Start with most recent

# ============================================================
# Functions to build and visualize MST
# ============================================================
# IMPORTANT: These functions build and display MST for ONE SPECIFIC window at a time.
# - The MST is built using ONLY the correlation matrix for the selected window
#   (e.g., 10-day, 30-day, or 50-day rolling correlation)
# - This is NOT an average across windows - each window has its own separate MST
# - Edge strengths show correlations for THAT SPECIFIC window and date only
# - Use arrow keys (← →) to switch between different windows and see how
#   the correlation structure changes with different time horizons
# ============================================================

def build_mst(window, date):
    """
    Build MST for a SINGLE specific window and date.
    
    Parameters:
    -----------
    window : int
        The rolling window size (e.g., 10, 30, or 50 days)
    date : datetime
        The specific date to analyze
    
    Returns:
    --------
    mst : networkx.Graph
        Minimum spanning tree for this window/date combination
    dist_matrix : pandas.DataFrame
        Distance matrix (converted from correlation)
    corr_matrix : pandas.DataFrame
        Correlation matrix for this specific window at this date
    
    Note:
    -----
    This builds the MST using ONLY the correlations from the specified window.
    For example, if window=30, this uses the 30-day rolling correlation on that date.
    Edge weights represent the correlation strength for THAT window, not averaged.
    """
    corr_df = rolling_corrs[window]
    corr_matrix = corr_df.loc[date]
    
    # Check for NaN values and fill them
    if corr_matrix.isnull().any().any():
        corr_matrix = corr_matrix.fillna(0)
    
    # Convert to distance and build MST
    corr_matrix_clipped = corr_matrix.clip(-1, 1)
    dist_matrix = np.sqrt(2 * (1 - corr_matrix_clipped))
    dist_matrix = dist_matrix.fillna(2.0)
    
    # Ensure diagonal is 0 (no self-loops)
    np.fill_diagonal(dist_matrix.values, 0.0)
    
    G = nx.from_pandas_adjacency(dist_matrix)
    
    # Remove any self-loops just in case
    G.remove_edges_from(nx.selfloop_edges(G))
    
    mst = nx.minimum_spanning_tree(G)
    
    return mst, dist_matrix, corr_matrix

def visualize_mst(mst, corr_matrix, window, date):
    """
    Visualize MST with dramatic edge thickness and color for ONE specific window/date.
    
    Parameters:
    -----------
    mst : networkx.Graph
        The minimum spanning tree to visualize
    corr_matrix : pandas.DataFrame
        Correlation matrix for this specific window and date
    window : int
        The rolling window size (10, 30, or 50 days)
    date : datetime
        The specific date being visualized
    
    Visualization Details:
    ----------------------
    - Edge thickness: Thicker = stronger correlation (cubic scaling for drama)
    - Edge color: Red = strong positive correlation, Blue = weak correlation
    - Statistics box: Shows top 3 strongest correlations for THIS window/date only
    - NOT averaged across windows - shows correlation structure at this specific
      time horizon and date
    
    Navigation:
    -----------
    Use arrow keys to explore different windows and dates to see how correlation
    structure changes over time and across different rolling window sizes.
    """
    ax.clear()
    
    # Position nodes (use consistent seed for stable layout)
    pos = nx.spring_layout(mst, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(mst, pos, 
                           node_color='#4A90E2',
                           node_size=2000,
                           alpha=0.95,
                           ax=ax)
    
    # Get edge weights and convert to correlation (stronger correlation = thicker edge)
    edges = list(mst.edges())
    distances = [mst[u][v]['weight'] for u, v in edges]
    
    # Convert distance back to correlation: corr = 1 - (dist^2)/2
    correlations = [1 - (d**2)/2 for d in distances]
    
    # EXTREMELY DRAMATIC edge thickness based on correlation strength
    # Scale from 0.3 (weak) to 35 (very strong) - much more dramatic!
    min_width = 0.3
    max_width = 35.0
    widths = []
    for corr in correlations:
        # Exponential scaling for dramatic effect
        # Use cubic power for even more dramatic differences
        normalized = max(0, corr)  # Ensure non-negative
        width = min_width + (max_width - min_width) * (normalized ** 3)
        widths.append(width)
    
    # Color gradient: red (strong correlation) to blue (weak correlation)
    colors = []
    for corr in correlations:
        # Map correlation to color: high correlation = red, low = blue
        normalized = max(0, min(1, corr))
        colors.append((1 - normalized, 0, normalized, 0.85))  # RGBA
    
    # Draw edges with dramatic thickness and color
    for (u, v), width, color in zip(edges, widths, colors):
        nx.draw_networkx_edges(mst, pos,
                               [(u, v)],
                               width=width,
                               edge_color=[color],
                               ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(mst, pos,
                           font_size=10,
                           font_weight='bold',
                           ax=ax)
    
    # Find strongest correlations and create statistics text
    # Get all pairwise correlations (upper triangle only)
    corr_values = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            ticker1 = corr_matrix.index[i]
            ticker2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            if not np.isnan(corr):
                corr_values.append((ticker1, ticker2, corr))
    
    # Sort by correlation strength (absolute value) and get top 3
    corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
    top_correlations = corr_values[:3]
    
    # Create statistics text box
    # Note: Showing correlations for THIS specific date's {window}-day rolling window
    stats_text = f"📊 {window}-day Rolling Correlation\n"
    stats_text += f"Date: {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)}\n"
    stats_text += f"\nStrongest Correlations (MST Edges):\n"
    for i, (t1, t2, corr) in enumerate(top_correlations[:3], 1):
        stats_text += f"{i}. {t1}-{t2}: {corr:.3f}\n"
    
    # Add text box in bottom-left corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom', bbox=props, 
            family='monospace')
    
    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
    
    # Calculate which date this is (out of how many total trading days)
    # Note: available_dates contains ALL dates in the dataset (2022-2024)
    # You can navigate through the entire ~750 trading days using arrow keys
    date_num = current_date_idx + 1  # Current position in dataset
    total_dates = len(available_dates)  # Total trading days in dataset
    
    ax.set_title(f'Minimum Spanning Tree - {window}-day Rolling Correlation\n'
                 f'Date: {date_str} (showing {date_num} of {total_dates} trading days)\n'
                 f'Edge Thickness/Color = Correlation Strength (Red/Thick=Strong, Blue/Thin=Weak)\n'
                 f'Use ← → to change window (10/30/50 days) | ↑ ↓ to change date',
                 fontsize=12, fontweight='bold', pad=20)
    ax.axis('off')
    
    fig.canvas.draw()

# ============================================================
# Event handler for arrow key navigation
# ============================================================
def on_key(event):
    global current_window_idx, current_date_idx
    
    if event.key == 'right':
        # Next window
        current_window_idx = (current_window_idx + 1) % len(windows)
        update_plot()
    elif event.key == 'left':
        # Previous window
        current_window_idx = (current_window_idx - 1) % len(windows)
        update_plot()
    elif event.key == 'up':
        # Next date (more recent)
        current_date_idx = min(current_date_idx + 1, len(available_dates) - 1)
        update_plot()
    elif event.key == 'down':
        # Previous date (older)
        current_date_idx = max(current_date_idx - 1, 0)
        update_plot()
    elif event.key in ['q', 'escape']:
        plt.close()

def update_plot():
    """
    Update plot with current window and date.
    
    Builds MST on-demand for the selected window/date combination.
    Does NOT pre-generate all MSTs - calculates only when needed for navigation.
    """
    global available_dates, current_date_idx
    
    window = windows[current_window_idx]
    
    # Update available dates for current window (ALL dates, not just last 10)
    all_dates = rolling_corrs[window].index.get_level_values(0).unique()
    available_dates = sorted(all_dates)  # All dates in dataset
    
    # Adjust date index if needed
    if current_date_idx >= len(available_dates):
        current_date_idx = len(available_dates) - 1
    
    date = available_dates[current_date_idx]
    
    # Build MST on-demand for this specific window and date
    mst, dist_matrix, corr_matrix = build_mst(window, date)
    visualize_mst(mst, corr_matrix, window, date)

# ============================================================
# Initial visualization
# ============================================================
print("\nControls:")
print("  ← → : Switch between 10/30/50-day windows")
print("  ↑ ↓ : Navigate through ALL dates in dataset (~750 trading days)")
print("  Q/Esc: Quit")

fig, ax = plt.subplots(figsize=(16, 12))
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot
update_plot()

print("\n✓ Showing MST visualization...")
print("Use arrow keys to navigate!")
plt.show()

print("✓ Done")

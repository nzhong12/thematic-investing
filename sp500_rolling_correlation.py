"""
Rolling Correlation Analysis with CRSP WRDS Data
Downloads daily adjusted close prices for 10 large S&P stocks and computes rolling correlations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('macosx')  # Use macOS native backend for arrow key navigation
import matplotlib.pyplot as plt
import seaborn as sns
import wrds

print("="*80)
print("ROLLING CORRELATION ANALYSIS - 10 LARGE S&P STOCKS")
print("="*80)

# ============================================================
# STEP 1: Configuration and pull S&P 500 stocks from WRDS
# ============================================================
print("\n" + "="*80)
print("STEP 1: Getting S&P 500 stocks from WRDS")
print("="*80)

start_date = "2024-01-01"  # Using 2024 data (2025 data may not be available yet)
end_date = "2024-12-31"
num_stocks = 10

print(f"\nDate range: {start_date} to {end_date}")
print(f"Number of stocks to select: {num_stocks}")

print("\nConnecting to WRDS...")
db = wrds.Connection()

# Query to get S&P 500 constituent tickers from CRSP
# Get stocks that were in S&P 500 during our date range
print("\n✓ Querying S&P 500 constituents...")
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
print(f"✓ Found {len(sp500_df)} S&P 500 stocks in CRSP")

# Select first N stocks with valid tickers
tickers = sp500_df['ticker'].dropna().unique()[:num_stocks].tolist()

print(f"\n✓ Selected {len(tickers)} stocks: {tickers}")

# ============================================================
# STEP 2: download data from WRDS CRSP
# ============================================================
print("\n" + "="*80)
print("STEP 2: Downloading daily adjusted close prices from WRDS CRSP")
print("="*80)

# query to get daily adjusted prices for specific tickers
# Using CRSP daily stock file (dsf) with price and returns
query = """
    SELECT 
        a.date,
        a.permno,
        b.ticker,
        a.prc,
        a.ret,
        a.cfacpr,
        a.cfacshr
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

print("Executing query...")
df = db.raw_sql(query)
db.close()

print(f"\n✓ Downloaded {len(df)} rows")
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
print(f"✓ Unique tickers: {sorted(df['ticker'].unique())}")

print("\n📊 Sample of raw data:")
print(df.head(10))

# ============================================================
# STEP 3: compute daily simple returns
# ============================================================
print("\n" + "="*80)
print("STEP 3: Computing daily simple returns")
print("="*80)

# convert to pivot format (dates × tickers) for prices
print("\n✓ Pivoting prices to wide format...")
prices = df.pivot(index='date', columns='ticker', values='prc')

# take absolute value of prices (CRSP uses negative prices for bid/ask averages)
prices = prices.abs()

print(f"✓ Prices shape: {prices.shape}")
print(f"\n📊 Sample prices:")
print(prices.head())

# compute daily returns using pct_change()
print("\n✓ Computing daily returns with pct_change()...")
returns = prices.pct_change().dropna()

print(f"\n✓ Returns shape: {returns.shape}")
print(f"  ({returns.shape[0]} trading days × {returns.shape[1]} stocks)")

print(f"\n📊 Sample returns:")
print(returns.head(10))

print(f"\n📊 Returns summary statistics:")
print(returns.describe())

# ============================================================
# STEP 4: Compute rolling correlations from daily returns
# ============================================================
print("\n" + "="*80)
print("STEP 4: Computing Rolling Correlation Matrices")
print("="*80)

print("""
CORRELATION COEFFICIENT EXPLANATION:
Each entry ρ_ij = Cov(returns_i, returns_j) / (σ_i * σ_j)
- ρ_ij = 1.0  → perfect positive correlation (stocks move together)
- ρ_ij = 0.0  → no linear correlation
- ρ_ij = -1.0 → perfect negative correlation (stocks move opposite)

Each matrix summarizes how the past 'window' days of returns co-moved.
""")

# Define rolling windows
windows = [10, 30, 50]
print(f"Rolling windows: {windows} days")
print(f"  - 10 days: ~2 trading weeks")
print(f"  - 30 days: ~1.5 trading months")
print(f"  - 50 days: ~2.5 trading months\n")

# Store correlation matrices for each window in a dictionary
# Structure: rolling_corrs = {10: DataFrame, 30: DataFrame, 50: DataFrame}
# Each DataFrame is MultiIndex(date, ticker) × tickers with ~251 correlation matrices
rolling_corrs = {}

# rolling_corrs = {
    # 10:  <DataFrame with 241×10 rows × 10 cols>,
    # 30:  <DataFrame with 221×10 rows × 10 cols>,
    # 50:  <DataFrame with 201×10 rows × 10 cols>
# }

for window in windows:
    print(f"\n{'='*80}")
    print(f"COMPUTING {window}-DAY ROLLING CORRELATIONS")
    print(f"{'='*80}")
    
    # compute rolling correlation using pandas
    # returns.rolling(window).corr() creates a multi-index DataFrame:
    # Level 0: date, Level 1: ticker, Columns: ticker
    print(f"\n✓ Computing rolling {window}-day correlation matrices...")
    rolling_corr = returns.rolling(window=window).corr()
    
    print(f"✓ Computed {len(rolling_corr.index.get_level_values(0).unique())} correlation matrices")
    print(f"✓ Data structure: MultiIndex(date, ticker) × tickers")
    
    # store for later use
    rolling_corrs[window] = rolling_corr # compute all matrices per window
    
    # Extract the most recent (latest date) correlation matrix
    latest_date = rolling_corr.index.get_level_values(0)[-1]
    latest_corr_matrix = rolling_corr.loc[latest_date]
    
    # Display the correlation matrix
    print(f"\n{'─'*80}")
    print(f"--- {window}-day rolling correlation matrix ---")
    print(f"Latest date: {latest_date}")
    print(f"{'─'*80}")
    print("\n📊 Correlation Matrix (rounded to 2 decimals):")
    print(latest_corr_matrix.round(2))
    
    # Show some interesting statistics
    print(f"\n📊 Correlation Statistics:")
    # Get upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(latest_corr_matrix, dtype=bool), k=1)
    upper_triangle = latest_corr_matrix.where(mask)
    corr_values = upper_triangle.stack().values
    
    print(f"  Mean correlation: {corr_values.mean():.4f}")
    print(f"  Median correlation: {np.median(corr_values):.4f}")
    print(f"  Min correlation: {corr_values.min():.4f}")
    print(f"  Max correlation: {corr_values.max():.4f}")
    print(f"  Std deviation: {corr_values.std():.4f}")
    
    # Find highest and lowest correlations
    print(f"\n📊 Highest correlations:")
    for i, ticker1 in enumerate(latest_corr_matrix.index):
        for j, ticker2 in enumerate(latest_corr_matrix.columns):
            if i < j:  # Upper triangle only
                corr_val = latest_corr_matrix.loc[ticker1, ticker2]
                if corr_val > 0.7:  # Strong positive correlation
                    print(f"  {ticker1} <-> {ticker2}: {corr_val:.4f}")
    
    print(f"\n📊 Lowest correlations:")
    for i, ticker1 in enumerate(latest_corr_matrix.index):
        for j, ticker2 in enumerate(latest_corr_matrix.columns):
            if i < j:  # Upper triangle only
                corr_val = latest_corr_matrix.loc[ticker1, ticker2]
                if corr_val < 0.3:  # Weak or negative correlation
                    print(f"  {ticker1} <-> {ticker2}: {corr_val:.4f}")
    
    # PROOF: Show that we have ALL matrices saved - display last 10 full matrices
    print(f"\n{'='*80}")
    print(f"PROOF: FULL CORRELATION MATRICES FOR LAST 10 TRADING DAYS")
    print(f"{'='*80}")
    
    # Get all unique dates
    all_dates = rolling_corr.index.get_level_values(0).unique()
    last_10_dates = all_dates[-10:]
    
    print(f"\n✓ Total matrices computed for this window: {len(all_dates)}")
    print(f"✓ Displaying FULL matrices for last 10 dates: {last_10_dates[0]} to {last_10_dates[-1]}")
    print(f"✓ Each matrix is {len(returns.columns)} × {len(returns.columns)}")
    
    # Display each full correlation matrix
    for matrix_idx, date in enumerate(last_10_dates, 1):
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        print(f"\n{'─'*80}")
        print(f"Matrix {matrix_idx}/10 - Date: {date_str} ({window}-day window)")
        print(f"{'─'*80}")
        
        corr_matrix = rolling_corr.loc[date]
        print(corr_matrix.round(2))
        
        # Quick stats for this date
        mask_proof = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_tri_proof = corr_matrix.where(mask_proof)
        corr_vals_proof = upper_tri_proof.stack().values
        print(f"  → Avg: {corr_vals_proof.mean():.4f}, Min: {corr_vals_proof.min():.4f}, Max: {corr_vals_proof.max():.4f}")

# ============================================================
# STEP 5: Visualize correlation matrices
# ============================================================
print("\n" + "="*80)
print("STEP 5: Visualizing Correlation Matrices")
print("="*80)

print("\n✓ Creating heatmap visualizations...")

# Create a figure with subplots for each window
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Rolling Correlation Matrices - Latest Date', fontsize=16, fontweight='bold')

for idx, window in enumerate(windows):
    # Get latest correlation matrix
    rolling_corr = rolling_corrs[window]
    latest_date = rolling_corr.index.get_level_values(0)[-1]
    latest_corr_matrix = rolling_corr.loc[latest_date]
    
    # Create heatmap
    ax = axes[idx]
    im = ax.imshow(latest_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(latest_corr_matrix.columns)))
    ax.set_yticks(range(len(latest_corr_matrix.index)))
    ax.set_xticklabels(latest_corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(latest_corr_matrix.index)
    
    # Add title
    date_str = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, 'strftime') else str(latest_date)
    ax.set_title(f'{window}-day Correlations\n{date_str}', 
                 fontsize=12, fontweight='bold')
    
    # Add correlation values as text
    for i in range(len(latest_corr_matrix.index)):
        for j in range(len(latest_corr_matrix.columns)):
            text = ax.text(j, i, f'{latest_corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

plt.tight_layout()
print(f"✓ Heatmaps created for windows: {windows}")
print("✓ Displaying plots...")
plt.show()

# ============================================================
# STEP 6: Interactive plot - Navigate through dates AND windows
# ============================================================
print("\n" + "="*80)
print("STEP 6: Interactive Correlation Matrix Navigator")
print("="*80)

print("\n✓ Creating interactive plot...")
print("  Use LEFT/RIGHT arrow keys to navigate through last 10 days")
print("  Use UP/DOWN arrow keys to change window length")
print("  Press 'q' to quit")

# Prepare data for all windows
all_dates_dict = {}
for window in windows:
    rolling_corr = rolling_corrs[window]
    all_dates = rolling_corr.index.get_level_values(0).unique()
    all_dates_dict[window] = all_dates[-10:]  # Last 10 dates for each window

# Create interactive figure
fig_interactive, ax_interactive = plt.subplots(figsize=(12, 9))

# State variable for current index and window
class Navigator:
    def __init__(self):
        self.current_date_idx = 9  # Start at most recent (index 9 of last 10)
        self.current_window_idx = 2  # Start at 50-day window (index 2 of [10, 30, 50])
        
navigator = Navigator()

def update_plot():
    """Update the heatmap with current date and window's correlation matrix."""
    ax_interactive.clear()
    
    # Get current window and date
    current_window = windows[navigator.current_window_idx]
    current_dates = all_dates_dict[current_window]
    current_date = current_dates[navigator.current_date_idx]
    
    # Get correlation matrix for current window and date
    rolling_corr = rolling_corrs[current_window]
    corr_matrix = rolling_corr.loc[current_date]
    
    # Create heatmap
    im = ax_interactive.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax_interactive.set_xticks(range(len(corr_matrix.columns)))
    ax_interactive.set_yticks(range(len(corr_matrix.index)))
    ax_interactive.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax_interactive.set_yticklabels(corr_matrix.index)
    
    # Add title with window, date, and position
    date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
    window_info = f'{current_window}-day window (↑↓ to change)'
    date_info = f'Date {navigator.current_date_idx + 1}/10: {date_str} (←→ to change)'
    ax_interactive.set_title(f'{window_info}\n{date_info}', 
                            fontsize=12, fontweight='bold', pad=20)
    
    # Add correlation values as text
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            text = ax_interactive.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar if it doesn't exist
    if not hasattr(update_plot, 'colorbar_added'):
        cbar = plt.colorbar(im, ax=ax_interactive)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        update_plot.colorbar_added = True
    
    # Add statistics text
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)
    corr_vals = upper_tri.stack().values
    
    stats_text = f'Window: {current_window}d | Avg: {corr_vals.mean():.3f} | Min: {corr_vals.min():.3f} | Max: {corr_vals.max():.3f}'
    fig_interactive.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig_interactive.canvas.draw()

def on_key(event):
    """Handle keyboard events for date and window navigation."""
    print(f"DEBUG: Key pressed: '{event.key}'")  # Debug output
    
    if event.key == 'right':
        # Move forward in time (more recent date)
        if navigator.current_date_idx < 9:  # 0-9 indices for 10 dates
            navigator.current_date_idx += 1
            current_window = windows[navigator.current_window_idx]
            current_date = all_dates_dict[current_window][navigator.current_date_idx]
            date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
            print(f"  → Moving to date {navigator.current_date_idx + 1}/10: {date_str}")
            update_plot()
        else:
            print("  → Already at most recent date")
            
    elif event.key == 'left':
        # Move backward in time (older date)
        if navigator.current_date_idx > 0:
            navigator.current_date_idx -= 1
            current_window = windows[navigator.current_window_idx]
            current_date = all_dates_dict[current_window][navigator.current_date_idx]
            date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
            print(f"  → Moving to date {navigator.current_date_idx + 1}/10: {date_str}")
            update_plot()
        else:
            print("  → Already at oldest date")
            
    elif event.key == 'up':
        # Increase window length (10 → 30 → 50)
        if navigator.current_window_idx < len(windows) - 1:
            navigator.current_window_idx += 1
            current_window = windows[navigator.current_window_idx]
            print(f"  ↑ Switched to {current_window}-day window")
            update_plot()
        else:
            print("  ↑ Already at longest window (50 days)")
            
    elif event.key == 'down':
        # Decrease window length (50 → 30 → 10)
        if navigator.current_window_idx > 0:
            navigator.current_window_idx -= 1
            current_window = windows[navigator.current_window_idx]
            print(f"  ↓ Switched to {current_window}-day window")
            update_plot()
        else:
            print("  ↓ Already at shortest window (10 days)")
            
    elif event.key == 'q':
        print("  → Closing interactive plot")
        plt.close(fig_interactive)

# Connect the event handler
cid = fig_interactive.canvas.mpl_connect('key_press_event', on_key)
print(f"\n✓ Event handler connected (ID: {cid})")

# Initial plot (starts at 50-day window, most recent date)
update_plot()

print("\n" + "="*80)
print("✓ Interactive plot ready!")
print("="*80)
print("\n📍 NAVIGATION CONTROLS:")
print("  ← LEFT arrow   : Go back in time (older dates)")
print("  → RIGHT arrow  : Go forward in time (newer dates)")
print("  ↑ UP arrow     : Increase window length (10→30→50 days)")
print("  ↓ DOWN arrow   : Decrease window length (50→30→10 days)")
print("  q              : Close the plot")
print("\n⚠️  IMPORTANT: Click on the plot window to give it keyboard focus first!")
print("\n📊 Current view: 50-day window, most recent date (2024-12-31)")
print(f"📊 Available windows: {windows}")
print(f"📊 Available dates: Last 10 trading days")
print("\nShowing interactive plot...")

plt.show(block=True)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ Successfully analyzed rolling correlations:")
print(f"   - Stocks: {len(tickers)} ({', '.join(tickers)})")
print(f"   - Trading days: {len(returns)}")
min_date_str = returns.index.min().strftime('%Y-%m-%d') if hasattr(returns.index.min(), 'strftime') else str(returns.index.min())
max_date_str = returns.index.max().strftime('%Y-%m-%d') if hasattr(returns.index.max(), 'strftime') else str(returns.index.max())
print(f"   - Date range: {min_date_str} to {max_date_str}")
print(f"   - Rolling windows: {windows} days")

print(f"\n✅ Key findings:")
for window in windows:
    rolling_corr = rolling_corrs[window]
    latest_date = rolling_corr.index.get_level_values(0)[-1]
    latest_corr_matrix = rolling_corr.loc[latest_date]
    
    # Get upper triangle
    mask = np.triu(np.ones_like(latest_corr_matrix, dtype=bool), k=1)
    upper_triangle = latest_corr_matrix.where(mask)
    corr_values = upper_triangle.stack().values
    
    print(f"   - {window}-day window: avg correlation = {corr_values.mean():.4f}")

print("\n" + "="*80)
print("Analysis complete! ✓")

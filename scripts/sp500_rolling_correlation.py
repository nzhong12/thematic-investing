"""
Rolling Correlation Analysis with CRSP WRDS Data
Downloads daily adjusted close prices for 10 large S&P stocks and computes rolling correlations.

EXPORTS for other scripts:
    - rolling_corrs: dict of {window: DataFrame} with all correlation matrices
    - tickers: list of stock tickers
    - returns: DataFrame of daily returns
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

start_date = "2022-01-01"  # Start Jan 1, 2022 for full 3-year analysis
end_date = "2024-12-31"    # End Dec 31, 2024
num_stocks = 20
print("NOTE: Selecting stocks with complete data history from 2022-2024 to ensure full 3-year analysis.")

print(f"\nDate range: {start_date} to {end_date}")
print(f"Number of stocks to select: {num_stocks}")

print("\nConnecting to WRDS...")
db = wrds.Connection()

# Query to get S&P 500 constituent tickers that were CONSISTENTLY in the index
# during the entire date range (to avoid missing data issues)
print("\n✓ Querying S&P 500 constituents with complete data history...")
sp500_query = """
    SELECT DISTINCT a.permno, b.ticker
    FROM crsp.dsp500list as a
    LEFT JOIN crsp.dsenames as b
    ON a.permno = b.permno
        AND b.namedt <= '{start_date}'
        AND b.nameendt >= '{end_date}'
    WHERE a.start <= '{start_date}'
        AND a.ending >= '{end_date}'
        AND b.ticker IS NOT NULL
    ORDER BY a.permno
    LIMIT 100
""".format(start_date=start_date, end_date=end_date)

sp500_df = db.raw_sql(sp500_query)
print(f"✓ Found {len(sp500_df)} S&P 500 stocks with complete data from 2022-2024")

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

# Store average correlations for each stock across all windows
avg_corrs_by_window = {}

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
    
    # Calculate average correlation for each stock across ALL dates in this window
    all_dates_window = rolling_corr.index.get_level_values(0).unique()
    stock_avg_corrs = {}
    
    for stock in latest_corr_matrix.index:
        # Get all correlation values for this stock across all dates (excluding self-correlation)
        all_corrs = []
        for date in all_dates_window:
            corr_matrix_date = rolling_corr.loc[date]
            stock_corrs = corr_matrix_date.loc[stock].drop(stock)  # Exclude self
            all_corrs.extend(stock_corrs.values)
        stock_avg_corrs[stock] = np.mean(all_corrs)
    
    avg_corrs_by_window[window] = stock_avg_corrs
    
    # Get all unique dates for analysis
    all_dates = rolling_corr.index.get_level_values(0).unique()
    
    # Display last 2 days correlation matrices for visual reference
    print(f"\n{'='*80}")
    print(f"CORRELATION MATRICES - LAST 2 TRADING DAYS")
    print(f"{'='*80}")
    
    last_2_dates = all_dates[-2:]
    print(f"\nShowing dates: {last_2_dates[0]} and {last_2_dates[-1]}")
    print(f"Window: {window} days")
    
    for day_idx, date in enumerate(last_2_dates, 1):
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        print(f"\n{'='*80}")
        print(f"CORRELATION MATRIX - {date_str} ({window}-day window)")
        print(f"{'='*80}")
        
        corr_matrix = rolling_corr.loc[date]
        print("\n", corr_matrix.round(2))
    
    # Save full time series of correlation matrices to outputs folder (all dates from 2022-01-01 to 2024-12-31)
    # 
    # CSV STRUCTURE EXPLANATION:
    # --------------------------
    # Each row represents ONE TRADING DATE
    # Columns:
    #   - 'date': The trading date
    #   - 'STOCK1-STOCK2': Correlation coefficient between the two stocks for that date
    #                      (e.g., 'AAPL-MSFT' = correlation between Apple and Microsoft)
    # 
    # Example row:
    #   date,AAPL-MSFT,AAPL-GOOGL,MSFT-GOOGL,...
    #   2024-01-15,0.7543,0.6821,0.8234,...
    # 
    # This means on 2024-01-15:
    #   - AAPL and MSFT had 0.7543 correlation (75.43% co-movement over past N days)
    #   - AAPL and GOOGL had 0.6821 correlation
    #   - etc.
    # 
    # The correlation is calculated using the past N days of returns (rolling window),
    # where N is the window size (10, 30, or 50 days).
    #
    print(f"\n✓ Preparing full time series CSV for {window}-day window...")
    print(f"  Each row = 1 trading date with all stock-pair correlations for that date")
    print(f"  Each column = 1 stock pair (e.g., 'AAPL-MSFT')")
    
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Restructure: each row is a date, columns are stock pairs (e.g., "AAPL-MSFT")
    time_series_data = []
    for date in all_dates:
        corr_matrix = rolling_corr.loc[date]
        row_data = {'date': date}
        
        # Add all unique pairs (upper triangle only to avoid duplicates)
        for i, stock1 in enumerate(corr_matrix.index):
            for j, stock2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle only
                    pair_name = f"{stock1}-{stock2}"
                    row_data[pair_name] = corr_matrix.loc[stock1, stock2]
        
        time_series_data.append(row_data)
    
    # Convert to DataFrame and save to outputs folder
    time_series_df = pd.DataFrame(time_series_data)
    time_series_csv = f'outputs/correlation_{window}day_2022-2024.csv'
    time_series_df.to_csv(time_series_csv, index=False)
    print(f"✓ Saved full time series ({len(all_dates)} dates) to: {time_series_csv}")
    
    # TOP 10 MOST CORRELATED PAIRS (by magnitude) - ACROSS ENTIRE YEAR
    print(f"\n{'='*80}")
    print(f"TOP 10 MOST CORRELATED PAIRS (by magnitude) - {window}-day window")
    print(f"{'='*80}")
    print(f"Analyzing average correlations across ALL {len(all_dates)} trading days")
    
    # Calculate average correlation for each pair across ALL dates
    pairs_avg_list = []
    
    # Get unique stock pairs
    latest_corr = rolling_corr.loc[all_dates[-1]]
    for i, stock1 in enumerate(latest_corr.index):
        for j, stock2 in enumerate(latest_corr.columns):
            if i < j:  # Upper triangle only to avoid duplicates
                # Calculate average correlation across all dates (excluding NaN)
                corr_values = []
                for date in all_dates:
                    corr_matrix = rolling_corr.loc[date]
                    val = corr_matrix.loc[stock1, stock2]
                    if not np.isnan(val):  # Skip NaN values
                        corr_values.append(val)
                
                # Only calculate if we have valid data
                if len(corr_values) > 0:
                    avg_corr = np.mean(corr_values)
                    std_corr = np.std(corr_values)
                    min_corr = np.min(corr_values)
                    max_corr = np.max(corr_values)
                    
                    pairs_avg_list.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'avg_correlation': avg_corr,
                        'std_correlation': std_corr,
                        'min_correlation': min_corr,
                        'max_correlation': max_corr,
                        'abs_avg_correlation': abs(avg_corr),
                        'num_valid_days': len(corr_values)
                    })
    
    # Sort by absolute value of average correlation and get top 10
    pairs_avg_df = pd.DataFrame(pairs_avg_list)
    
    if len(pairs_avg_df) == 0:
        print("\n⚠️  No valid correlation data available for this window")
        print(f"   (This can happen if window size {window} is too large for the data)")
    else:
        top_10_pairs = pairs_avg_df.nlargest(10, 'abs_avg_correlation')
        
        print(f"\n{'Rank':<6}{'Stock 1':<8}{'Stock 2':<8}{'Avg Corr':>12}{'Std Dev':>10}{'Range':>18}")
        print(f"{'─'*6}{'─'*8}{'─'*8}{'─'*12}{'─'*10}{'─'*18}")
        for idx, row in enumerate(top_10_pairs.itertuples(), 1):
            corr_range = f"[{row.min_correlation:+.2f}, {row.max_correlation:+.2f}]"
            print(f"{idx:<6}{row.stock1:<8}{row.stock2:<8}{row.avg_correlation:>+12.4f}{row.std_correlation:>10.4f}{corr_range:>18}")
        
        # Save top 10 pairs to file
        output_file = f'outputs/top10_correlations_{window}day.txt'
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"TOP 10 MOST CORRELATED PAIRS - {window}-day Rolling Window\n")
            f.write("="*80 + "\n")
            f.write(f"\nAnalysis Period: 2022-01-01 to 2024-12-31\n")
            f.write(f"Total Trading Days Analyzed: {len(all_dates)}\n")
            f.write(f"Rolling Window: {window} days\n")
            f.write(f"\nMetrics Explanation:\n")
            f.write(f"  • Avg Corr: Average correlation over all {len(all_dates)} trading days\n")
            f.write(f"  • Std Dev: Standard deviation (volatility) of the correlation\n")
            f.write(f"  • Range: [Min, Max] correlation values observed\n")
            f.write("\n" + "="*80 + "\n\n")
            
            f.write(f"{'Rank':<6}{'Stock 1':<10}{'Stock 2':<10}{'Avg Corr':>12}{'Std Dev':>12}{'Range':>20}\n")
            f.write("─"*80 + "\n")
            for idx, row in enumerate(top_10_pairs.itertuples(), 1):
                corr_range = f"[{row.min_correlation:+.2f}, {row.max_correlation:+.2f}]"
                f.write(f"{idx:<6}{row.stock1:<10}{row.stock2:<10}{row.avg_correlation:>+12.4f}{row.std_correlation:>12.4f}{corr_range:>20}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*80 + "\n\n")
            f.write("High Average Correlation (>0.70):\n")
            f.write("  → Stocks tend to move together consistently over time\n")
            f.write("  → May belong to same sector or have similar business models\n")
            f.write("  → Good candidates for sector rotation strategies\n\n")
            f.write("Moderate Correlation (0.40-0.70):\n")
            f.write("  → Stocks show meaningful but not perfect co-movement\n")
            f.write("  → May share some common factors but retain independence\n")
            f.write("  → Useful for diversification within a theme\n\n")
            f.write("Low Standard Deviation (<0.15):\n")
            f.write("  → Correlation is stable over time\n")
            f.write("  → Relationship is persistent and reliable\n")
            f.write("  → Better for long-term strategic positioning\n\n")
            f.write("High Standard Deviation (>0.20):\n")
            f.write("  → Correlation varies significantly over time\n")
            f.write("  → Relationship may be regime-dependent\n")
            f.write("  → Requires active monitoring and tactical adjustments\n\n")
            f.write("="*80 + "\n")
            f.write("NOTE: Full correlation time series data available in:\n")
            f.write(f"      outputs/correlation_{window}day_2022-2024.csv\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Saved top 10 pairs analysis to: {output_file}")
    
    # PERSISTENT HIGH CORRELATIONS ANALYSIS
    print(f"\n{'='*80}")
    print(f"PERSISTENT HIGH CORRELATIONS - {window}-day window")
    print(f"{'='*80}")
    print(f"Checking which pairs have stayed highly correlated (|r| > 0.60) over time")
    
    # Define "high correlation" threshold and "persistent" period
    HIGH_CORR_THRESHOLD = 0.60
    LOOKBACK_DAYS = min(30, len(all_dates))  # Look back 30 days or all available
    
    print(f"\nCriteria:")
    print(f"  - High correlation: |correlation| > {HIGH_CORR_THRESHOLD}")
    print(f"  - Persistent period: Last {LOOKBACK_DAYS} trading days")
    print(f"  - Consistency: High correlation in ≥80% of days checked")
    
    # Check each pair over the lookback period
    lookback_dates = all_dates[-LOOKBACK_DAYS:]
    persistent_pairs = []
    
    for i, stock1 in enumerate(latest_corr.index):
        for j, stock2 in enumerate(latest_corr.columns):
            if i < j:  # Upper triangle only
                # Count how many days this pair had high correlation (skip NaN)
                high_corr_days = 0
                total_days = 0
                corr_values_for_avg = []
                
                for check_date in lookback_dates:
                    corr_matrix_check = rolling_corr.loc[check_date]
                    corr_val = corr_matrix_check.loc[stock1, stock2]
                    
                    # Skip NaN values
                    if not np.isnan(corr_val):
                        total_days += 1
                        corr_values_for_avg.append(corr_val)
                        if abs(corr_val) > HIGH_CORR_THRESHOLD:
                            high_corr_days += 1
                
                # If high correlation in ≥80% of days, it's persistent
                consistency = high_corr_days / total_days if total_days > 0 else 0
                if consistency >= 0.80 and len(corr_values_for_avg) > 0:
                    avg_corr = np.mean(corr_values_for_avg)
                    persistent_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'avg_correlation': avg_corr,
                        'consistency_pct': consistency * 100,
                        'days_high': high_corr_days,
                        'total_days': total_days
                    })
    
    if len(persistent_pairs) > 0:
        persistent_df = pd.DataFrame(persistent_pairs).sort_values('consistency_pct', ascending=False)
        print(f"\n✓ Found {len(persistent_pairs)} persistently correlated pairs:\n")
        print(f"{'Stock 1':<8}{'Stock 2':<8}{'Avg Corr':>12}{'Consistency':>12}{'Days':>10}")
        print(f"{'─'*8}{'─'*8}{'─'*12}{'─'*12}{'─'*10}")
        for row in persistent_df.itertuples():
            print(f"{row.stock1:<8}{row.stock2:<8}{row.avg_correlation:>+12.4f}{row.consistency_pct:>11.1f}%"
                  f"{row.days_high:>5}/{row.total_days:<3}")
    else:
        print(f"\n✗ No pairs found with persistent high correlation over {LOOKBACK_DAYS} days")
        print(f"   (This is common for longer windows like 50-day)")
    
    print(f"\n{'='*80}")

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

# Save data for MST visualization
import pickle
print("\nSaving correlation data for MST visualization...")
with open('correlation_data.pkl', 'wb') as f:
    pickle.dump({
        'rolling_corrs': rolling_corrs,
        'tickers': tickers,
        'returns': returns
    }, f)
print("✓ Saved to correlation_data.pkl")

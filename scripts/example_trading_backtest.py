"""
Example: Backtest Theme-Based Trading Strategies

This script demonstrates how to generate trading signals based on CLUSTER DYNAMICS:

WHAT ARE CLUSTER DYNAMICS?
--------------------------
Cluster dynamics refer to how stocks move between different thematic groups over time:
  • Stocks ENTERING clusters (joining a theme)
  • Stocks LEAVING clusters (exiting a theme)
  • Clusters GROWING in size (theme strengthening)
  • Clusters SHRINKING in size (theme weakening)
  • Stocks DIVERGING from their cluster's average return

WHY DO THESE DYNAMICS MATTER FOR TRADING?
------------------------------------------
1. Theme Formation: When stocks enter large, persistent clusters → capital flows into theme
2. Theme Dissolution: When stocks leave clusters → capital exits, momentum weakens
3. Within-Theme Arbitrage: Stocks diverging from cluster mean tend to revert
4. Theme Rotation: Growing clusters attract capital from shrinking clusters

WHAT THIS SCRIPT DOES:
---------------------
1. Load historical cluster membership data (which stocks were in which clusters each day)
2. Generate 3 types of trading signals based on different cluster dynamics:
   - Momentum: Trade cluster entry/exit ("theme joining/leaving")
   - Mean Reversion: Trade within-cluster divergence ("outlier correction")
   - Rotation: Trade cluster size trends ("theme strengthening/weakening")
3. Backtest strategies with realistic constraints (commission, position limits)
4. Evaluate performance vs. buy-and-hold benchmark
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pgi_theme_graphs.trading_strategy import ClusterSignalGenerator, SimpleBacktester


def load_cluster_data(window='30day'):
    """
    Load cluster data from outputs.
    
    This data tells us WHICH stocks were in WHICH clusters on EACH day.
    By tracking changes over time, we can detect cluster dynamics:
      - If AAPL moves from cluster A to cluster B → cluster transition
      - If cluster size grows from 5 to 15 stocks → theme strengthening
      - If cluster persists 50+ days → stable theme (high quality signal)
    
    Returns:
    --------
    pd.DataFrame with columns: date, ticker, cluster_id, cluster_size
      - date: Trading date
      - ticker: Stock symbol
      - cluster_id: Unique cluster identifier (includes date)
      - cluster_size: Number of stocks in that cluster on that date
    """
    output_dir = Path(__file__).parent / 'outputs'
    
    # Load Jaccard clustering results
    cluster_file = output_dir / f'jaccard_clusters_{window}_2022-2024.txt'
    
    if not cluster_file.exists():
        print(f"Error: {cluster_file} not found!")
        print("Please run scripts/extract_clusters_jaccard.py first.")
        return None
    
    # Parse cluster file
    # Format: 2023-01-04 [1 clusters] | cluster1_tickers | cluster2_tickers | ...
    records = []
    
    with open(cluster_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip header lines and empty lines
            if not line or line.startswith('=') or line.startswith('Format:') or \
               line.startswith('Period:') or line.startswith('Total') or \
               line.startswith('Correlation') or line.startswith('Clustering:') or \
               line.startswith('Note:') or line.startswith('SUMMARY') or \
               line.startswith('•') or line.startswith('Top ') or \
               line.startswith('DAILY') or line.startswith('MOST') or \
               line.startswith('(within') or '(' in line and ')' in line and 'size=' in line:
                continue
            
            # Parse data lines: "2023-01-04 [1 clusters] | AAPL,ABBV,... | ..."
            if '[' in line and 'clusters]' in line and '|' in line:
                try:
                    # Extract date
                    date_str = line.split('[')[0].strip()
                    current_date = pd.to_datetime(date_str)
                    
                    # Extract clusters (split by |, skip the first part with date)
                    parts = line.split('|')
                    clusters = [p.strip() for p in parts[1:] if p.strip()]
                    
                    # Process each cluster
                    for cluster_num, cluster_stocks in enumerate(clusters, 1):
                        tickers = [t.strip() for t in cluster_stocks.split(',')]
                        cluster_size = len(tickers)
                        
                        # Create records
                        for ticker in tickers:
                            if ticker:  # Skip empty strings
                                records.append({
                                    'date': current_date,
                                    'ticker': ticker,
                                    'cluster_id': f"{current_date.strftime('%Y%m%d')}_C{cluster_num}",
                                    'cluster_size': cluster_size
                                })
                except Exception as e:
                    print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                    continue
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} cluster membership records from {cluster_file.name}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique stocks: {df['ticker'].nunique()}")
    
    return df


def load_price_data(window='30day'):
    """
    Load correlation data and extract price information.
    
    For simplicity, we'll use correlation data as proxy.
    In production, you'd want actual OHLC data.
    """
    output_dir = Path(__file__).parent / 'outputs'
    corr_file = output_dir / f'correlation_{window}_2022-2024.csv'
    
    if not corr_file.exists():
        print(f"Error: {corr_file} not found!")
        return None
    
    # Load correlation data
    corr_df = pd.read_csv(corr_file)
    
    # For this example, we'll generate synthetic prices
    # In production, load actual prices from WRDS
    print("\nNote: Using synthetic price data for demonstration.")
    print("For production backtesting, load actual OHLC prices from WRDS.")
    
    # Get dates and extract tickers from column names (e.g., "AAPL-ABBV" -> ["AAPL", "ABBV"])
    dates = pd.to_datetime(corr_df['date'].unique())
    ticker_pairs = [col for col in corr_df.columns if '-' in col and col != 'date']
    tickers = sorted(set([t for pair in ticker_pairs for t in pair.split('-')]))
    
    # Generate synthetic prices (random walk)
    np.random.seed(42)
    prices = pd.DataFrame(
        index=dates,
        columns=tickers,
        data=100 * np.exp(np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.02)
    )
    
    return prices


def run_backtest_example():
    """
    Run a complete backtest example.
    """
    print("=" * 70)
    print("THEME-BASED TRADING STRATEGY BACKTEST")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading cluster data...")
    cluster_data = load_cluster_data(window='30day')
    
    if cluster_data is None:
        return
    
    print("\n[2/5] Loading price data...")
    price_data = load_price_data(window='30day')
    
    if price_data is None:
        return
    
    # Initialize signal generator
    print("\n[3/5] Initializing signal generator...")
    print("\nThe signal generator will analyze cluster dynamics to create trading signals:")
    print("  • CLUSTER MOMENTUM: Detect when stocks join/leave strong themes")
    print("  • MEAN REVERSION: Detect when stocks diverge from their cluster's behavior")
    print("  • THEME ROTATION: Detect when themes are growing vs. shrinking")
    print("")
    
    signal_gen = ClusterSignalGenerator(
        cluster_data=cluster_data,
        price_data=price_data,
        lookback_window=20  # Look back 20 days to compute trends and statistics
    )
    
    print(f"  - Identified {len(signal_gen.cluster_persistence)} unique clusters")
    print(f"  - Detected {len(signal_gen.cluster_transitions)} cluster transitions")
    
    # Initialize backtester
    print("\n[4/5] Running backtest...")
    backtester = SimpleBacktester(
        initial_capital=100000,
        commission=0.001,  # 10 bps
        max_positions=20
    )
    
    # Run backtest day-by-day
    dates = sorted(cluster_data['date'].unique())
    
    print("\nHow signals work on each trading day:")
    print("  1. Check which clusters each stock belongs to (cluster membership)")
    print("  2. Detect if stocks changed clusters vs. yesterday (momentum signal)")
    print("  3. Check if stocks deviated from their cluster's average return (mean reversion)")
    print("  4. Measure if clusters are growing or shrinking (rotation signal)")
    print("  5. Combine all 3 signals → final buy/sell decision\n")
    
    for i, date in enumerate(dates):
        if date not in price_data.index:
            continue
        
        # Generate combined signals based on cluster dynamics
        # Weights determine how much each type of dynamic matters:
        #   40% cluster entry/exit (momentum)
        #   30% within-cluster divergence (mean reversion)
        #   30% cluster size trends (rotation)
        signals = signal_gen.combine_signals(
            date=date,
            weights={'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3}
        )
        
        # Execute trades
        prices = price_data.loc[date]
        backtester.execute_signals(
            date=date,
            signals=signals,
            prices=prices,
            signal_threshold=0.5
        )
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  - Processed {i + 1}/{len(dates)} days...")
    
    # Performance analysis
    print("\n[5/5] Performance Analysis")
    print("=" * 70)
    
    stats = backtester.get_performance_stats()
    
    print(f"\nStrategy Performance:")
    print(f"  Total Return:     {stats['total_return']:>8.2f}%")
    print(f"  Sharpe Ratio:     {stats['sharpe_ratio']:>8.2f}")
    print(f"  Max Drawdown:     {stats['max_drawdown']:>8.2f}%")
    print(f"  Number of Trades: {stats['num_trades']:>8.0f}")
    print(f"  Avg Positions:    {stats['avg_positions']:>8.1f}")
    print(f"  Final Equity:     ${stats['final_equity']:>8,.0f}")
    
    # Benchmark (equal-weight buy-and-hold)
    print("\nBenchmark (Equal-Weight Buy-and-Hold):")
    initial_price = price_data.iloc[0].mean()
    final_price = price_data.iloc[-1].mean()
    benchmark_return = (final_price / initial_price - 1) * 100
    print(f"  Total Return:     {benchmark_return:>8.2f}%")
    
    # Excess return
    excess = stats['total_return'] - benchmark_return
    print(f"\nExcess Return:      {excess:>8.2f}%")
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    
    output_dir = Path(__file__).parent / 'outputs'
    
    # Equity curve
    equity_df = pd.DataFrame(backtester.equity_curve)
    equity_df.to_csv(output_dir / 'backtest_equity_curve.csv', index=False)
    print(f"  - Saved equity curve to: backtest_equity_curve.csv")
    
    # Trade log
    trades_df = pd.DataFrame(backtester.trades)
    trades_df.to_csv(output_dir / 'backtest_trades.csv', index=False)
    print(f"  - Saved trade log to: backtest_trades.csv")
    
    print("\n✓ Backtest complete!")
    print("=" * 70)


def analyze_signal_distribution():
    """
    Analyze signal distribution without running full backtest.
    
    This shows you WHAT the cluster dynamics look like on a specific day:
      - Which stocks are entering/leaving clusters?
      - Which stocks diverged from their cluster's behavior?
      - Which clusters are growing vs. shrinking?
      - What are the final combined buy/sell signals?
    
    Useful for understanding WHY certain signals are generated.
    """
    print("\n" + "=" * 70)
    print("SIGNAL DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    cluster_data = load_cluster_data(window='30day')
    price_data = load_price_data(window='30day')
    
    if cluster_data is None or price_data is None:
        return
    
    signal_gen = ClusterSignalGenerator(
        cluster_data=cluster_data,
        price_data=price_data,
        lookback_window=20
    )
    
    # Analyze signals for last date
    last_date = cluster_data['date'].max()
    
    print(f"\nAnalyzing signals for: {last_date.strftime('%Y-%m-%d')}")
    print("-" * 70)
    
    # Individual strategies - each captures different cluster dynamics
    momentum = signal_gen.cluster_momentum_signal(last_date)
    mean_rev = signal_gen.cluster_mean_reversion_signal(last_date)
    rotation = signal_gen.theme_rotation_signal(last_date)
    
    print(f"\nCluster Momentum Signals (based on cluster entry/exit):")
    print(f"  • Stocks entering strong clusters (theme formation):  {sum(1 for v in momentum.values() if v > 0)} BUY")
    print(f"  • Stocks leaving clusters or entering weak clusters:  {sum(1 for v in momentum.values() if v < 0)} SELL")
    print(f"  → Logic: Capital flows into strengthening themes")
    
    print(f"\nMean Reversion Signals (based on within-cluster divergence):")
    print(f"  • Stocks underperforming their cluster's average:  {sum(1 for v in mean_rev.values() if v > 0)} BUY")
    print(f"  • Stocks outperforming their cluster's average:    {sum(1 for v in mean_rev.values() if v < 0)} SELL")
    print(f"  → Logic: Correlated stocks revert to cluster mean")
    
    print(f"\nTheme Rotation Signals (based on cluster size trends):")
    print(f"  • Stocks in growing clusters (theme strengthening): {sum(1 for v in rotation.values() if v > 0)} BUY")
    print(f"  • Stocks in shrinking clusters (theme weakening):   {sum(1 for v in rotation.values() if v < 0)} SELL")
    print(f"  → Logic: Rotate from dying themes to emerging themes")
    
    # Combined signals
    combined = signal_gen.combine_signals(last_date)
    
    print(f"\nCombined Signals (Top 10 Buys):")
    print(f"Signal = 0.4×Momentum + 0.3×MeanReversion + 0.3×Rotation")
    print("-" * 70)
    print(f"  Ticker  Signal   (Mom=entry/exit | MR=divergence | Rot=growth)")
    print("-" * 70)
    top_buys = combined.nlargest(10, 'signal')
    for _, row in top_buys.iterrows():
        print(f"  {row['ticker']:>6s}  {row['signal']:>5.2f}   "
              f"(Mom: {row['momentum']:>4.1f} | MR: {row['mean_reversion']:>4.1f} | "
              f"Rot: {row['rotation']:>4.1f})")
    
    print(f"\nCombined Signals (Top 10 Sells):")
    print("-" * 70)
    top_sells = combined.nsmallest(10, 'signal')
    for _, row in top_sells.iterrows():
        print(f"  {row['ticker']:>6s}  Signal: {row['signal']:>5.2f}  "
              f"(Mom: {row['momentum']:>4.1f}, MR: {row['mean_reversion']:>4.1f}, "
              f"Rot: {row['rotation']:>4.1f})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Theme-based trading strategy backtest')
    parser.add_argument('--mode', choices=['backtest', 'signals'], default='backtest',
                       help='Mode: backtest (run full backtest) or signals (analyze signal distribution)')
    
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        run_backtest_example()
    else:
        analyze_signal_distribution()

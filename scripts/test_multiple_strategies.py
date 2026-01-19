"""
Test Multiple Trading Strategy Variations

Try different configurations to find positive returns:
1. Reverse signals (inverse strategy)
2. Different signal weights
3. Momentum only (no mean reversion)
4. Long-only (no selling)
5. Different cluster size thresholds
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pgi_theme_graphs.trading_strategy import ClusterSignalGenerator, SimpleBacktester


def load_cluster_data(window='30day'):
    """Load cluster data from outputs."""
    output_dir = Path(__file__).parent / 'outputs'
    cluster_file = output_dir / f'jaccard_clusters_{window}_2022-2024.txt'
    
    if not cluster_file.exists():
        print(f"Error: {cluster_file} not found!")
        return None
    
    records = []
    with open(cluster_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('=') or line.startswith('Format:') or \
               line.startswith('Period:') or line.startswith('Total') or \
               line.startswith('Correlation') or line.startswith('Clustering:') or \
               line.startswith('Note:') or line.startswith('SUMMARY') or \
               line.startswith('•') or line.startswith('Top ') or \
               line.startswith('DAILY') or line.startswith('MOST') or \
               line.startswith('(within') or '(' in line and ')' in line and 'size=' in line:
                continue
            
            if '[' in line and 'clusters]' in line and '|' in line:
                try:
                    date_str = line.split('[')[0].strip()
                    current_date = pd.to_datetime(date_str)
                    
                    parts = line.split('|')
                    clusters = [p.strip() for p in parts[1:] if p.strip()]
                    
                    for cluster_num, cluster_stocks in enumerate(clusters, 1):
                        tickers = [t.strip() for t in cluster_stocks.split(',')]
                        cluster_size = len(tickers)
                        
                        for ticker in tickers:
                            if ticker:
                                records.append({
                                    'date': current_date,
                                    'ticker': ticker,
                                    'cluster_id': f"{current_date.strftime('%Y%m%d')}_C{cluster_num}",
                                    'cluster_size': cluster_size
                                })
                except Exception as e:
                    continue
    
    return pd.DataFrame(records)


def load_price_data(window='30day'):
    """Load price data."""
    output_dir = Path(__file__).parent / 'outputs'
    corr_file = output_dir / f'correlation_{window}_2022-2024.csv'
    
    if not corr_file.exists():
        return None
    
    corr_df = pd.read_csv(corr_file)
    dates = pd.to_datetime(corr_df['date'].unique())
    ticker_pairs = [col for col in corr_df.columns if '-' in col and col != 'date']
    tickers = sorted(set([t for pair in ticker_pairs for t in pair.split('-')]))
    
    # Synthetic prices
    np.random.seed(42)
    prices = pd.DataFrame(
        index=dates,
        columns=tickers,
        data=100 * np.exp(np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.02)
    )
    
    return prices


def run_strategy_test(cluster_data, price_data, config):
    """
    Run backtest with specific configuration.
    
    config: dict with keys:
        - name: str
        - weights: dict
        - signal_threshold: float
        - min_cluster_size: int
        - max_cluster_size: int
        - reverse_signals: bool
        - long_only: bool
    """
    signal_gen = ClusterSignalGenerator(
        cluster_data=cluster_data,
        price_data=price_data,
        lookback_window=20
    )
    
    backtester = SimpleBacktester(
        initial_capital=100000,
        commission=0.001,
        max_positions=20
    )
    
    dates = sorted(cluster_data['date'].unique())
    
    for date in dates:
        if date not in price_data.index:
            continue
        
        signals = signal_gen.combine_signals(
            date=date,
            weights=config['weights'],
            min_cluster_size=config['min_cluster_size'],
            max_cluster_size=config['max_cluster_size']
        )
        
        # Reverse signals if configured
        if config.get('reverse_signals', False):
            signals['signal'] = -signals['signal']
            signals['momentum'] = -signals['momentum']
            signals['mean_reversion'] = -signals['mean_reversion']
            signals['rotation'] = -signals['rotation']
        
        # Long-only: zero out negative signals
        if config.get('long_only', False):
            signals.loc[signals['signal'] < 0, 'signal'] = 0
        
        prices = price_data.loc[date]
        backtester.execute_signals(
            date=date,
            signals=signals,
            prices=prices,
            signal_threshold=config['signal_threshold']
        )
    
    stats = backtester.get_performance_stats()
    return stats


def main():
    print("=" * 70)
    print("TESTING MULTIPLE STRATEGY VARIATIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    cluster_data = load_cluster_data(window='30day')
    price_data = load_price_data(window='30day')
    
    if cluster_data is None or price_data is None:
        print("Error loading data!")
        return
    
    # Calculate benchmark
    initial_price = price_data.iloc[0].mean()
    final_price = price_data.iloc[-1].mean()
    benchmark_return = (final_price / initial_price - 1) * 100
    
    print(f"Benchmark (Buy-and-Hold): {benchmark_return:.2f}%\n")
    print("=" * 70)
    
    # Define test configurations
    configs = [
        # Original
        {
            'name': 'V2: Balanced (Current)',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.5,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': False,
            'long_only': False
        },
        # Try reversing signals
        {
            'name': 'V3: REVERSED Signals',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.5,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': True,
            'long_only': False
        },
        # Momentum only
        {
            'name': 'V4: Momentum Only',
            'weights': {'momentum': 1.0, 'mean_reversion': 0.0, 'rotation': 0.0},
            'signal_threshold': 0.5,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': False,
            'long_only': False
        },
        # Momentum only REVERSED
        {
            'name': 'V5: Momentum Only REVERSED',
            'weights': {'momentum': 1.0, 'mean_reversion': 0.0, 'rotation': 0.0},
            'signal_threshold': 0.5,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': True,
            'long_only': False
        },
        # Long only (no selling)
        {
            'name': 'V6: Long Only',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.3,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': False,
            'long_only': True
        },
        # Mean reversion only
        {
            'name': 'V7: Mean Reversion Only',
            'weights': {'momentum': 0.0, 'mean_reversion': 1.0, 'rotation': 0.0},
            'signal_threshold': 0.5,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': False,
            'long_only': False
        },
        # Mean reversion REVERSED
        {
            'name': 'V8: Mean Reversion REVERSED',
            'weights': {'momentum': 0.0, 'mean_reversion': 1.0, 'rotation': 0.0},
            'signal_threshold': 0.5,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': True,
            'long_only': False
        },
        # Larger clusters
        {
            'name': 'V9: Larger Clusters (5-50)',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.5,
            'min_cluster_size': 5,
            'max_cluster_size': 50,
            'reverse_signals': False,
            'long_only': False
        },
        # Smaller clusters
        {
            'name': 'V10: Smaller Clusters (3-20)',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.5,
            'min_cluster_size': 3,
            'max_cluster_size': 20,
            'reverse_signals': False,
            'long_only': False
        },
        # Lower threshold (more trades)
        {
            'name': 'V11: Lower Threshold (0.3)',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.3,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': False,
            'long_only': False
        },
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config['name']}")
        print("-" * 70)
        
        stats = run_strategy_test(cluster_data, price_data, config)
        
        print(f"  Return:       {stats['total_return']:>8.2f}%")
        print(f"  Sharpe:       {stats['sharpe_ratio']:>8.2f}")
        print(f"  Max DD:       {stats['max_drawdown']:>8.2f}%")
        print(f"  Trades:       {stats['num_trades']:>8.0f}")
        print(f"  Final Equity: ${stats['final_equity']:>8,.0f}")
        
        results.append({
            'name': config['name'],
            'return': stats['total_return'],
            'sharpe': stats['sharpe_ratio'],
            'max_dd': stats['max_drawdown'],
            'trades': stats['num_trades'],
            'final_equity': stats['final_equity']
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - RANKED BY RETURN")
    print("=" * 70)
    print(f"{'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'Final $':>12}")
    print("-" * 70)
    
    results_df = pd.DataFrame(results).sort_values('return', ascending=False)
    for _, row in results_df.iterrows():
        color = "✅" if row['return'] > 0 else "❌"
        print(f"{color} {row['name']:<33} {row['return']:>9.2f}% {row['sharpe']:>8.2f} {row['trades']:>8.0f} ${row['final_equity']:>10,.0f}")
    
    print("\n" + "=" * 70)
    print(f"Benchmark: {benchmark_return:.2f}%")
    print("=" * 70)
    
    # Save results
    results_df.to_csv(Path(__file__).parent / 'outputs' / 'strategy_comparison.csv', index=False)
    print("\nSaved results to: strategy_comparison.csv")


if __name__ == "__main__":
    main()

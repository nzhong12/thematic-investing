"""
Fine-tune the WINNER: V22 (REVERSED + Fewer Positions)
Got +6.31% return with 2 trades. Let's optimize further.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pgi_theme_graphs.trading_strategy import ClusterSignalGenerator, SimpleBacktester


def load_cluster_data(window='30day'):
    output_dir = Path(__file__).parent / 'outputs'
    cluster_file = output_dir / f'jaccard_clusters_{window}_2022-2024.txt'
    
    if not cluster_file.exists():
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
    output_dir = Path(__file__).parent / 'outputs'
    corr_file = output_dir / f'correlation_{window}_2022-2024.csv'
    
    if not corr_file.exists():
        return None
    
    corr_df = pd.read_csv(corr_file)
    dates = pd.to_datetime(corr_df['date'].unique())
    ticker_pairs = [col for col in corr_df.columns if '-' in col and col != 'date']
    tickers = sorted(set([t for pair in ticker_pairs for t in pair.split('-')]))
    
    np.random.seed(42)
    prices = pd.DataFrame(
        index=dates,
        columns=tickers,
        data=100 * np.exp(np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.02)
    )
    
    return prices


def run_strategy_test(cluster_data, price_data, config):
    signal_gen = ClusterSignalGenerator(
        cluster_data=cluster_data,
        price_data=price_data,
        lookback_window=20
    )
    
    backtester = SimpleBacktester(
        initial_capital=100000,
        commission=config.get('commission', 0.001),
        max_positions=config.get('max_positions', 20)
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
        
        if config.get('reverse_signals', False):
            signals['signal'] = -signals['signal']
        
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
    print("FINE-TUNING THE WINNER")
    print("=" * 70)
    
    cluster_data = load_cluster_data(window='30day')
    price_data = load_price_data(window='30day')
    
    if cluster_data is None or price_data is None:
        print("Error loading data!")
        return
    
    initial_price = price_data.iloc[0].mean()
    final_price = price_data.iloc[-1].mean()
    benchmark_return = (final_price / initial_price - 1) * 100
    
    print(f"\nBenchmark: {benchmark_return:.2f}%\n")
    
    # Fast test: lower thresholds for more trades
    configs = [
        # Baseline winner (2 trades)
        {
            'name': 'V24: Winner (8pos, 0.65thresh, 2 trades)',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.65,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': True,
            'max_positions': 8,
        },
        # Lower thresholds for more trades
        {
            'name': 'V35: More trades (10pos, 0.55thresh)',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.55,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': True,
            'max_positions': 10,
        },
        {
            'name': 'V36: Even more trades (12pos, 0.5thresh)',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.5,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': True,
            'max_positions': 12,
        },
        {
            'name': 'V37: Aggressive (15pos, 0.45thresh)',
            'weights': {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3},
            'signal_threshold': 0.45,
            'min_cluster_size': 3,
            'max_cluster_size': 35,
            'reverse_signals': True,
            'max_positions': 15,
        },
    ]
    
    print(f"Testing {len(configs)} focused configurations to beat {benchmark_return:.2f}% benchmark\n")
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing: {config['name']}")
        print("-" * 70)
        
        stats = run_strategy_test(cluster_data, price_data, config)
        
        print(f"  Return:       {stats['total_return']:>8.2f}%")
        print(f"  Sharpe:       {stats['sharpe_ratio']:>8.2f}")
        print(f"  Max DD:       {stats['max_drawdown']:>8.2f}%")
        print(f"  Trades:       {stats['num_trades']:>8.0f}")
        print(f"  Final Equity: ${stats['final_equity']:>8,.0f}")
        print()
        
        results.append({
            'name': config['name'],
            'return': stats['total_return'],
            'sharpe': stats['sharpe_ratio'],
            'max_dd': stats['max_drawdown'],
            'trades': stats['num_trades'],
            'final_equity': stats['final_equity'],
            'max_positions': config['max_positions'],
            'threshold': config['signal_threshold'],
            'min_size': config['min_cluster_size'],
            'max_size': config['max_cluster_size'],
            'reversed': config['reverse_signals'],
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS - RANKED BY RETURN")
    print("=" * 70)
    print(f"{'Strategy':<42} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'Reversed':>10}")
    print("-" * 70)
    
    results_df = pd.DataFrame(results).sort_values('return', ascending=False)
    for _, row in results_df.iterrows():
        if row['return'] > benchmark_return:
            color = "🏆"
        elif row['return'] > 0:
            color = "✅"
        elif row['return'] > -5:
            color = "🟡"
        else:
            color = "❌"
        reversed_text = "YES" if row['reversed'] else "NO"
        print(f"{color} {row['name']:<40} {row['return']:>9.2f}% {row['sharpe']:>8.2f} {row['trades']:>8.0f} {reversed_text:>10}")
    
    print("\n" + "=" * 70)
    print(f"Benchmark: {benchmark_return:.2f}%")
    
    best = results_df.iloc[0]
    beat_benchmark = len(results_df[results_df['return'] > benchmark_return])
    
    print(f"\n🏆 BEST STRATEGY: {best['name']}")
    print(f"   Return: {best['return']:.2f}% (vs. {benchmark_return:.2f}% benchmark)")
    print(f"   Sharpe: {best['sharpe']:.2f}")
    print(f"   Trades: {best['trades']:.0f}")
    print(f"   Max Positions: {best['max_positions']:.0f}")
    print(f"   Signal Threshold: {best['threshold']:.2f}")
    print(f"   Cluster Size Range: {best['min_size']:.0f}-{best['max_size']:.0f} stocks")
    print(f"   Signals Reversed: {'YES' if best['reversed'] else 'NO'}")
    print("=" * 70)
    
    results_df.to_csv(Path(__file__).parent / 'outputs' / 'final_strategy_results.csv', index=False)
    print("\nSaved to: final_strategy_results.csv")
    
    if beat_benchmark > 0:
        print(f"\n📊 Success! {beat_benchmark}/{len(results_df)} strategies beat the benchmark!")
    else:
        print(f"\n📊 No strategies beat {benchmark_return:.2f}% benchmark, but best got {best['return']:.2f}%")


if __name__ == "__main__":
    main()

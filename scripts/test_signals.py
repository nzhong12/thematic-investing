import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from pgi_theme_graphs.trading_strategy import ClusterSignalGenerator

# Load cluster data
output_dir = Path(__file__).parent / 'outputs'
cluster_file = output_dir / 'jaccard_clusters_30day_2022-2024.txt'

records = []
with open(cluster_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or any(x in line for x in ['=', 'Format:', 'Period:', 'Total', 'SUMMARY', '•', 'Top', 'DAILY', 'MOST', '(within']):
            continue
        if '[' in line and 'clusters]' in line and '|' in line:
            try:
                date_str = line.split('[')[0].strip()
                current_date = pd.to_datetime(date_str)
                parts = line.split('|')
                clusters = [p.strip() for p in parts[1:] if p.strip()]
                for cluster_num, cluster_stocks in enumerate(clusters, 1):
                    tickers = [t.strip() for t in cluster_stocks.split(',')]
                    for ticker in tickers:
                        if ticker:
                            records.append({
                                'date': current_date,
                                'ticker': ticker,
                                'cluster_id': cluster_num,
                                'cluster_size': len(tickers),
                            })
            except:
                pass
        if len(records) > 5000:
            break

cluster_data = pd.DataFrame(records)
print(f'Loaded {len(cluster_data)} records')

# Create price data
corr_file = output_dir / 'correlation_30day_2022-2024.csv'
corr_df = pd.read_csv(corr_file)
dates = pd.to_datetime(corr_df['date'].unique())
ticker_pairs = [col for col in corr_df.columns if '-' in col and col != 'date']
tickers = sorted(set([t for pair in ticker_pairs for t in pair.split('-')]))

np.random.seed(42)
price_data = pd.DataFrame(
    index=dates,
    columns=tickers,
    data=100 * np.exp(np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.02)
)

# Test signal generation
sig_gen = ClusterSignalGenerator(cluster_data, price_data, 20)
test_date = dates[100]
print(f'\nTesting with date: {test_date}')

result = sig_gen.combine_signals(
    test_date, 
    {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3}, 
    3, 35
)

print(f'Result type: {type(result)}')
print(f'Result shape: {result.shape if hasattr(result, "shape") else "N/A"}')
print(f'Result columns: {result.columns.tolist() if hasattr(result, "columns") else "N/A"}')
print(f'\nFirst few rows:')
print(result.head(10))

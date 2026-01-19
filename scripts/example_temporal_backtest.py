"""
Backtest Trading Strategy Based on Temporal Graph Clusters (TGC)

This uses the temporal_edges.csv and temporal_nodes.csv to trade based on:
- Cluster persistence (edge weights)
- Emerging themes (new clusters)
- Dying themes (dissolving clusters)
- Theme strength changes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import wrds

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pgi_theme_graphs.temporal_trading_strategy import TemporalGraphTrader


def main():
    print("=" * 80)
    print("TEMPORAL GRAPH CLUSTER (TGC) TRADING STRATEGY")
    print("=" * 80)
    
    # File paths
    output_dir = Path(__file__).parent / 'outputs'
    edges_file = output_dir / 'temporal_edges.csv'
    nodes_file = output_dir / 'temporal_nodes.csv'
    
    # Check files exist
    if not edges_file.exists():
        print(f"\n❌ Error: {edges_file} not found!")
        print("Please run scripts/temporal_clusters.py first to generate temporal graph data.")
        return
    
    if not nodes_file.exists():
        print(f"\n❌ Error: {nodes_file} not found!")
        print("Please run scripts/temporal_clusters.py first to generate temporal graph data.")
        return
    
    # Initialize trader
    print("\n[1/4] Initializing temporal graph trader...")
    trader = TemporalGraphTrader(
        edges_file=str(edges_file),
        nodes_file=str(nodes_file)
    )
    
    # Load price data (from WRDS or fallback to yfinance)
    print("\n[2/4] Loading price data...")
    dates = sorted(trader.cluster_metrics['date'].unique())
    tickers = sorted(trader.cluster_membership['ticker'].unique())
    
    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')
    
    print(f"Date range: {start_date} to {end_date}")
    print(f"Tickers: {len(tickers)}")
    
    try:
        print("Attempting to connect to WRDS...")
        db = wrds.Connection()
        
        # Query daily prices from CRSP
        query = """
            SELECT 
                a.date,
                b.ticker,
                a.prc
            FROM crsp.dsf AS a
            INNER JOIN crsp.dsenames AS b
                ON a.permno = b.permno
                AND b.namedt <= a.date
                AND a.date <= b.nameendt
            WHERE b.ticker IN ({ticker_list})
                AND a.date >= '{start_date}'
                AND a.date <= '{end_date}'
                AND a.prc IS NOT NULL
            ORDER BY a.date, b.ticker
        """.format(
            start_date=start_date,
            end_date=end_date,
            ticker_list=','.join(f"'{t}'" for t in tickers)
        )
        
        print("Fetching prices from WRDS CRSP...")
        df = db.raw_sql(query)
        db.close()
        
        print(f"✓ Downloaded {len(df)} price records from WRDS")
        
        # Pivot to wide format and take absolute value
        prices = df.pivot(index='date', columns='ticker', values='prc').abs()
        
        # Ensure index is datetime
        prices.index = pd.to_datetime(prices.index)
        
        # Forward fill missing values (up to 5 days)
        price_data = prices.ffill(limit=5)
        
        print(f"✓ Price data shape: {price_data.shape}\n")
        
    except Exception as e:
        print(f"⚠️  WRDS connection failed: {str(e)[:100]}")
        print("\nFalling back to Yahoo Finance for price data...")
        
        try:
            import yfinance as yf
            
            # Download data for all tickers
            print(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            # Extract adjusted close prices
            if len(tickers) == 1:
                price_data = pd.DataFrame({tickers[0]: data['Adj Close']})
            else:
                price_data = data['Adj Close']
            
            # Ensure index is datetime
            price_data.index = pd.to_datetime(price_data.index)
            
            # Forward fill missing values
            price_data = price_data.ffill(limit=5)
            
            print(f"✓ Downloaded prices from Yahoo Finance")
            print(f"✓ Price data shape: {price_data.shape}\n")
            
        except ImportError:
            print("\n❌ yfinance not installed. Install with: pip install yfinance")
            print("Or fix WRDS connection to use real data.")
            print("\nUsing synthetic prices as fallback...\n")
            
            # Fallback to synthetic
            price_data = pd.DataFrame(
                index=dates,
                columns=tickers,
                data=100 * np.exp(np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.02)
            )
        except Exception as yf_error:
            print(f"❌ Yahoo Finance also failed: {str(yf_error)[:100]}")
            print("\nUsing synthetic prices as fallback...\n")
            
            # Fallback to synthetic
            np.random.seed(42)
            price_data = pd.DataFrame(
                index=dates,
                columns=tickers,
                data=100 * np.exp(np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.02)
            )
    
    # Show sample signals for most recent date
    print("\n[3/4] Sample Signals (Most Recent Date)")
    print("=" * 80)
    latest_date = dates[-1]
    print(f"\nDate: {latest_date.strftime('%Y-%m-%d')}\n")
    
    signals = trader.generate_signals(latest_date)
    
    # Sort by signal strength
    sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    
    print("Top 15 Signals:")
    print("-" * 80)
    print(f"{'Ticker':<8} {'Signal':<10} {'Reason'}")
    print("-" * 80)
    
    for ticker, signal in sorted_signals:
        direction = "BUY " if signal > 0 else "SELL"
        explanation = trader.get_signal_explanation(latest_date, ticker)
        print(f"{ticker:<8} {signal:>6.2f} {direction:<5} {explanation}")
    
    # Run backtest
    print("\n\n[4/4] Running Backtest")
    print("=" * 80)
    
    stats = trader.backtest(
        price_data=price_data,
        initial_capital=100000,
        signal_threshold=0.3,
        max_positions=20
    )
    
    # Check if backtest returned valid stats
    if not stats or 'total_return' not in stats:
        print("\n❌ Backtest failed to generate results!")
        print("This could mean:")
        print("  - No matching dates between cluster data and price data")
        print("  - No signals met the threshold criteria")
        print("  - No trades were executed")
        return
    
    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)
    
    print(f"\nTemporal Graph Strategy:")
    print(f"  Total Return:     {stats['total_return']:>8.2f}%")
    print(f"  Sharpe Ratio:     {stats['sharpe_ratio']:>8.2f}")
    print(f"  Max Drawdown:     {stats['max_drawdown']:>8.2f}%")
    print(f"  Number of Trades: {stats['num_trades']:>8.0f}")
    print(f"  Avg Positions:    {stats['avg_positions']:>8.1f}")
    print(f"  Final Equity:     ${stats['final_equity']:>8,.0f}")
    
    # Benchmark
    initial_price = price_data.iloc[0].mean()
    final_price = price_data.iloc[-1].mean()
    benchmark_return = (final_price / initial_price - 1) * 100
    
    print(f"\nBenchmark (Equal-Weight Buy-and-Hold):")
    print(f"  Total Return:     {benchmark_return:>8.2f}%")
    
    excess = stats['total_return'] - benchmark_return
    print(f"\nExcess Return:      {excess:>8.2f}%")
    
    print("\n" + "=" * 80)
    print("✓ Backtest complete!")
    print("=" * 80)
    
    print("\n💡 Strategy Logic:")
    print("  • BUY stocks in persistent themes (high edge weights)")
    print("  • BUY stocks in emerging themes (new large clusters)")
    print("  • SELL stocks in dying themes (no outgoing edges)")
    print("  • BUY stocks in strengthening themes (growing edge weights)")
    print("  • SELL stocks in weakening themes (declining edge weights)")
    
    print("\n📊 Key Metrics Used:")
    print("  • Edge weights = number of stocks shared between clusters")
    print("  • Persistence = average of incoming + outgoing edge weights")
    print("  • Emerging = new cluster with no incoming edges")
    print("  • Dying = cluster with no outgoing edges")
    
    if 'wrds' in str(type(price_data.iloc[0, 0]).__module__):
        data_source = "real WRDS/CRSP prices"
    else:
        data_source = "price data"
    
    print(f"\n✓ Backtest used {data_source} for {len(price_data)} trading days.")
    print(f"  Initial capital: $100,000")
    print(f"  Commission: 0.1% per trade")
    print(f"  Max positions: 20")
    print(f"  Signal threshold: 0.3")


if __name__ == "__main__":
    main()

"""
Trading Strategy Module for Theme-Based Investing

This module implements systematic trading strategies based on discovered stock clusters.
Key strategies:
1. Cluster momentum - Buy stocks entering strong/persistent clusters
2. Cluster mean reversion - Trade within-cluster divergence
3. Theme rotation - Rotate from weakening to strengthening themes
4. Cluster breakout - Trade stocks leaving/entering clusters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict


class ClusterSignalGenerator:
    """
    Generate trading signals based on cluster membership and dynamics.
    """
    
    def __init__(self, 
                 cluster_data: pd.DataFrame,
                 price_data: pd.DataFrame,
                 lookback_window: int = 20):
        """
        Initialize signal generator.
        
        Parameters:
        -----------
        cluster_data : pd.DataFrame
            Columns: ['date', 'ticker', 'cluster_id', 'cluster_size']
        price_data : pd.DataFrame
            Index: dates, Columns: tickers (price levels or returns)
        lookback_window : int
            Days to look back for signal calculations
        """
        self.cluster_data = cluster_data.sort_values('date')
        self.price_data = price_data
        self.lookback_window = lookback_window
        
        # Precompute cluster statistics
        self._compute_cluster_stats()
        
    def _compute_cluster_stats(self):
        """Compute cluster persistence, size, and stability metrics."""
        # Cluster persistence: how many days each cluster_id appears
        cluster_counts = self.cluster_data.groupby('cluster_id').size()
        self.cluster_persistence = cluster_counts.to_dict()
        
        # Average cluster size over time
        cluster_avg_size = self.cluster_data.groupby('cluster_id')['cluster_size'].mean()
        self.cluster_avg_size = cluster_avg_size.to_dict()
        
        # Cluster membership changes (stocks changing clusters)
        self.cluster_transitions = self._compute_transitions()
        
    def _compute_transitions(self) -> pd.DataFrame:
        """
        Compute cluster transitions for each stock.
        Returns DataFrame with columns: date, ticker, old_cluster, new_cluster
        """
        transitions = []
        
        for ticker in self.cluster_data['ticker'].unique():
            ticker_data = self.cluster_data[self.cluster_data['ticker'] == ticker].sort_values('date')
            ticker_data['prev_cluster'] = ticker_data['cluster_id'].shift(1)
            ticker_data['cluster_changed'] = ticker_data['cluster_id'] != ticker_data['prev_cluster']
            
            # Store transitions
            changed = ticker_data[ticker_data['cluster_changed'] == True].copy()
            if len(changed) > 0:
                transitions.append(changed[['date', 'ticker', 'prev_cluster', 'cluster_id']])
        
        if transitions:
            return pd.concat(transitions, ignore_index=True)
        else:
            return pd.DataFrame(columns=['date', 'ticker', 'prev_cluster', 'cluster_id'])
    
    def cluster_momentum_signal(self, date: pd.Timestamp, 
                                   min_cluster_size: int = 3,
                                   max_cluster_size: int = 35) -> Dict[str, float]:
        """
        Strategy 1: Cluster Momentum
        
        Logic:
        - Buy stocks that just entered believable, persistent themes (3-35 stocks)
        - Avoid mega-clusters (>35 stocks) - likely represents broad market regime
        - Sell stocks that left persistent themes or entered noise clusters
        
        Parameters:
        -----------
        min_cluster_size : int
            Minimum cluster size to consider (default: 3)
        max_cluster_size : int
            Maximum cluster size to consider (default: 35)
        
        Returns:
        --------
        Dict[ticker, signal] where signal in [-1, 1]
        """
        signals = {}
        
        # Get transitions on this date
        transitions_today = self.cluster_transitions[
            self.cluster_transitions['date'] == date
        ]
        
        for _, row in transitions_today.iterrows():
            ticker = row['ticker']
            new_cluster = row['cluster_id']
            old_cluster = row['prev_cluster']
            
            # Skip if NaN
            if pd.isna(old_cluster):
                continue
            
            # Get cluster characteristics
            new_persistence = self.cluster_persistence.get(new_cluster, 0)
            old_persistence = self.cluster_persistence.get(old_cluster, 0)
            new_size = self.cluster_avg_size.get(new_cluster, 0)
            old_size = self.cluster_avg_size.get(old_cluster, 0)
            
            # Filter: only trade believable themes (not mega-clusters, not tiny noise)
            new_is_believable = min_cluster_size <= new_size <= max_cluster_size
            old_is_believable = min_cluster_size <= old_size <= max_cluster_size
            
            # Strong buy: entering persistent, believable theme
            if new_is_believable and new_persistence > 30 and new_size >= 8:
                signals[ticker] = 1.0
            
            # Weak buy: entering moderately persistent, believable theme
            elif new_is_believable and new_persistence > 15 and new_size >= 5:
                signals[ticker] = 0.5
            
            # Sell: leaving believable persistent cluster
            elif old_is_believable and old_persistence > 20 and new_persistence < 10:
                signals[ticker] = -1.0
            
            # Weak sell: entering noise cluster or mega-cluster
            elif new_size < min_cluster_size or new_size > max_cluster_size:
                signals[ticker] = -0.5
        
        return signals
    
    def cluster_mean_reversion_signal(self, date: pd.Timestamp,
                                        min_cluster_size: int = 3,
                                        max_cluster_size: int = 35) -> Dict[str, float]:
        """
        Strategy 2: Cluster Mean Reversion
        
        Logic:
        - Within each believable cluster (3-35 stocks), identify divergence
        - Buy underperformers, sell outperformers (mean reversion within theme)
        - Skip mega-clusters (>35) - too broad for meaningful mean reversion
        
        Parameters:
        -----------
        min_cluster_size : int
            Minimum cluster size to consider (default: 3)
        max_cluster_size : int
            Maximum cluster size to consider (default: 35)
        
        Returns:
        --------
        Dict[ticker, signal] where signal in [-1, 1]
        """
        signals = {}
        
        # Get clusters on this date
        clusters_today = self.cluster_data[self.cluster_data['date'] == date]
        
        # Need at least lookback_window days of price history
        lookback_start = date - timedelta(days=self.lookback_window)
        
        if date not in self.price_data.index:
            return signals
        
        # Get returns over lookback period
        price_slice = self.price_data.loc[:date].tail(self.lookback_window)
        returns = price_slice.pct_change().dropna()
        
        # For each cluster, compute relative performance
        for cluster_id in clusters_today['cluster_id'].unique():
            cluster_members = clusters_today[
                clusters_today['cluster_id'] == cluster_id
            ]['ticker'].tolist()
            
            # Filter: only trade believable themes
            if len(cluster_members) < min_cluster_size or len(cluster_members) > max_cluster_size:
                continue
            
            # Get available returns for cluster members
            available_members = [t for t in cluster_members if t in returns.columns]
            if len(available_members) < 3:
                continue
            
            # Compute cumulative returns over lookback
            cluster_returns = returns[available_members].sum()  # Cumulative returns
            cluster_mean = cluster_returns.mean()
            cluster_std = cluster_returns.std()
            
            if cluster_std == 0:
                continue
            
            # Z-score: how many std devs each stock is from cluster mean
            for ticker in available_members:
                z_score = (cluster_returns[ticker] - cluster_mean) / cluster_std
                
                # Strong buy: significantly underperformed cluster
                if z_score < -1.5:
                    signals[ticker] = 1.0
                elif z_score < -1.0:
                    signals[ticker] = 0.5
                
                # Strong sell: significantly outperformed cluster
                elif z_score > 1.5:
                    signals[ticker] = -1.0
                elif z_score > 1.0:
                    signals[ticker] = -0.5
        
        return signals
    
    def theme_rotation_signal(self, date: pd.Timestamp,
                              min_cluster_size: int = 3,
                              max_cluster_size: int = 35) -> Dict[str, float]:
        """
        Strategy 3: Theme Rotation
        
        Logic:
        - Identify strengthening believable themes (3-35 stocks growing)
        - Identify weakening believable themes (shrinking)
        - Rotate from weak to strong themes
        - Ignore mega-clusters (>35) - too broad for theme-based trading
        
        Parameters:
        -----------
        min_cluster_size : int
            Minimum cluster size to consider (default: 3)
        max_cluster_size : int
            Maximum cluster size to consider (default: 35)
        
        Returns:
        --------
        Dict[ticker, signal] where signal in [-1, 1]
        """
        signals = {}
        
        # Get historical cluster sizes over lookback window
        lookback_start = date - timedelta(days=self.lookback_window)
        
        historical = self.cluster_data[
            (self.cluster_data['date'] > lookback_start) & 
            (self.cluster_data['date'] <= date)
        ]
        
        # Compute cluster size trends
        cluster_size_trend = historical.groupby('cluster_id').apply(
            lambda x: self._compute_size_trend(x)
        ).to_dict()
        
        # Get today's clusters
        clusters_today = self.cluster_data[self.cluster_data['date'] == date]
        
        for cluster_id in clusters_today['cluster_id'].unique():
            trend = cluster_size_trend.get(cluster_id, 0)
            persistence = self.cluster_persistence.get(cluster_id, 0)
            cluster_size = self.cluster_avg_size.get(cluster_id, 0)
            
            cluster_members = clusters_today[
                clusters_today['cluster_id'] == cluster_id
            ]['ticker'].tolist()
            
            # Filter: only trade believable themes
            if cluster_size < min_cluster_size or cluster_size > max_cluster_size:
                continue
            
            # Strengthening theme: growing + persistent + believable size
            if trend > 0.2 and persistence > 15:  # 20% growth, >15 days
                for ticker in cluster_members:
                    signals[ticker] = 1.0
            
            # Emerging theme: small but growing fast
            elif trend > 0.5 and persistence > 5:
                for ticker in cluster_members:
                    signals[ticker] = 0.5
            
            # Weakening theme: shrinking + low persistence
            elif trend < -0.2 and persistence < 10:
                for ticker in cluster_members:
                    signals[ticker] = -1.0
        
        return signals
    
    def _compute_size_trend(self, cluster_history: pd.DataFrame) -> float:
        """
        Compute linear trend of cluster size over time.
        Returns slope (% change per day)
        """
        if len(cluster_history) < 2:
            return 0.0
        
        sizes = cluster_history.sort_values('date')['cluster_size'].values
        x = np.arange(len(sizes))
        
        # Linear regression: y = mx + b
        if len(x) > 1:
            slope = np.polyfit(x, sizes, 1)[0]
            return slope / sizes.mean() if sizes.mean() > 0 else 0.0
        
        return 0.0
    
    def combine_signals(self, 
                       date: pd.Timestamp,
                       weights: Optional[Dict[str, float]] = None,
                       min_cluster_size: int = 3,
                       max_cluster_size: int = 35) -> pd.DataFrame:
        """
        Combine multiple signals with custom weights.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Date to generate signals for
        weights : Dict[str, float]
            Weights for each strategy: {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3}
        min_cluster_size : int
            Minimum cluster size to consider (default: 3)
        max_cluster_size : int
            Maximum cluster size to consider (default: 35)
        
        Returns:
        --------
        pd.DataFrame with columns: ticker, signal, momentum, mean_reversion, rotation
        """
        if weights is None:
            weights = {'momentum': 0.4, 'mean_reversion': 0.3, 'rotation': 0.3}
        
        # Generate individual signals with cluster size filters
        momentum = self.cluster_momentum_signal(date, min_cluster_size, max_cluster_size)
        mean_rev = self.cluster_mean_reversion_signal(date, min_cluster_size, max_cluster_size)
        rotation = self.theme_rotation_signal(date, min_cluster_size, max_cluster_size)
        
        # Combine all tickers
        all_tickers = set(momentum.keys()) | set(mean_rev.keys()) | set(rotation.keys())
        
        combined = []
        for ticker in all_tickers:
            mom_sig = momentum.get(ticker, 0)
            mr_sig = mean_rev.get(ticker, 0)
            rot_sig = rotation.get(ticker, 0)
            
            # Weighted combination
            final_signal = (
                mom_sig * weights['momentum'] +
                mr_sig * weights['mean_reversion'] +
                rot_sig * weights['rotation']
            )
            
            combined.append({
                'ticker': ticker,
                'signal': final_signal,
                'momentum': mom_sig,
                'mean_reversion': mr_sig,
                'rotation': rot_sig
            })
        
        # Handle case where no signals were generated
        if not combined:
            return pd.DataFrame(columns=['ticker', 'signal', 'momentum', 'mean_reversion', 'rotation'])
        
        return pd.DataFrame(combined).sort_values('signal', ascending=False)


class SimpleBacktester:
    """
    Simple backtester for theme-based strategies.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,  # 10 bps per trade
                 max_positions: int = 20):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital ($)
        commission : float
            Commission rate (fraction, e.g., 0.001 = 0.1%)
        max_positions : int
            Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_positions = max_positions
        
        self.portfolio = {}  # {ticker: shares}
        self.cash = initial_capital
        self.equity_curve = []
        self.trades = []
        
    def execute_signals(self, 
                       date: pd.Timestamp,
                       signals: pd.DataFrame,
                       prices: pd.Series,
                       signal_threshold: float = 0.5):
        """
        Execute trading signals on given date.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Current date
        signals : pd.DataFrame
            Output from SignalGenerator.combine_signals()
        prices : pd.Series
            Current prices {ticker: price}
        signal_threshold : float
            Minimum absolute signal strength to trade
        """
        # Filter strong signals
        strong_signals = signals[abs(signals['signal']) >= signal_threshold].copy()
        
        # Separate buys and sells
        buys = strong_signals[strong_signals['signal'] > 0].sort_values('signal', ascending=False)
        sells = strong_signals[strong_signals['signal'] < 0]
        
        # First, execute sells (free up cash)
        for _, row in sells.iterrows():
            ticker = row['ticker']
            if ticker in self.portfolio:
                self._close_position(date, ticker, prices[ticker], "SELL_SIGNAL")
        
        # Then, execute buys (up to max_positions)
        available_slots = self.max_positions - len(self.portfolio)
        
        for i, row in buys.iterrows():
            if available_slots <= 0:
                break
            
            ticker = row['ticker']
            signal_strength = row['signal']
            
            if ticker not in prices:
                continue
            
            price = prices[ticker]
            
            # Position size: stronger signal = larger position
            position_size = (self.cash / available_slots) * signal_strength
            shares = int(position_size / price)
            
            if shares > 0:
                self._open_position(date, ticker, price, shares, signal_strength)
                available_slots -= 1
        
        # Mark-to-market
        self._update_equity(date, prices)
    
    def _open_position(self, date, ticker, price, shares, signal):
        """Open new position."""
        cost = shares * price * (1 + self.commission)
        
        if cost > self.cash:
            return  # Not enough cash
        
        self.portfolio[ticker] = shares
        self.cash -= cost
        
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'value': cost,
            'signal': signal
        })
    
    def _close_position(self, date, ticker, price, reason):
        """Close existing position."""
        if ticker not in self.portfolio:
            return
        
        shares = self.portfolio[ticker]
        proceeds = shares * price * (1 - self.commission)
        
        self.cash += proceeds
        del self.portfolio[ticker]
        
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'value': proceeds,
            'signal': reason
        })
    
    def _update_equity(self, date, prices):
        """Update equity curve."""
        portfolio_value = sum(
            self.portfolio.get(ticker, 0) * prices.get(ticker, 0)
            for ticker in self.portfolio
        )
        
        total_equity = self.cash + portfolio_value
        
        self.equity_curve.append({
            'date': date,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'total_equity': total_equity,
            'num_positions': len(self.portfolio)
        })
    
    def get_performance_stats(self) -> Dict:
        """Compute performance statistics."""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        final_equity = equity_df['total_equity'].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # Daily returns
        equity_df['returns'] = equity_df['total_equity'].pct_change()
        
        sharpe = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
        max_dd = self._compute_max_drawdown(equity_df['total_equity'])
        
        trades_df = pd.DataFrame(self.trades)
        num_trades = len(trades_df)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': num_trades,
            'final_equity': final_equity,
            'avg_positions': equity_df['num_positions'].mean()
        }
    
    def _compute_max_drawdown(self, equity_series):
        """Compute maximum drawdown."""
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        return drawdown.min() * 100


if __name__ == "__main__":
    print("Theme-Based Trading Strategy Module")
    print("====================================")
    print("\nAvailable Strategies:")
    print("1. Cluster Momentum - Trade cluster entry/exit")
    print("2. Cluster Mean Reversion - Trade within-cluster divergence")
    print("3. Theme Rotation - Rotate from weak to strong themes")
    print("\nSee example_trading_backtest.py for usage examples.")

"""
Temporal Graph Cluster (TGC) Trading Strategy

Uses temporal_edges.csv and temporal_nodes.csv to generate trading signals based on:
1. Cluster persistence (edge weight trends)
2. Emerging themes (new clusters forming)
3. Dissolving themes (clusters disappearing)
4. Theme strength (cluster size + edge weights)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class TemporalGraphTrader:
    """
    Trade based on temporal graph cluster dynamics.
    
    Key signals:
    - PERSISTENCE: Buy stocks in clusters with high edge weights (strong persistence)
    - EMERGENCE: Buy stocks in newly forming clusters (theme birth)
    - DISSOLUTION: Sell stocks in clusters with declining edge weights (theme death)
    - STRENGTH: Buy stocks in large clusters with high average edge weights
    """
    
    def __init__(self, 
                 edges_file: str,
                 nodes_file: str):
        """
        Initialize with temporal graph data.
        
        Parameters:
        -----------
        edges_file : str
            Path to temporal_edges.csv
        nodes_file : str
            Path to temporal_nodes.csv  
        """
        print("Loading temporal graph data...")
        
        # Load edges (cluster transitions)
        self.edges = pd.read_csv(edges_file)
        self.edges['date_from'] = pd.to_datetime(self.edges['date_from'])
        self.edges['date_to'] = pd.to_datetime(self.edges['date_to'])
        
        # Load nodes (cluster properties)
        self.nodes = pd.read_csv(nodes_file)
        self.nodes['date'] = pd.to_datetime(self.nodes['date'])
        
        # Join edges with nodes to get cluster info
        # Add source cluster info
        self.edges = self.edges.merge(
            self.nodes[['node_id', 'date', 'cluster_idx']].rename(
                columns={'node_id': 'source', 'date': 'source_date', 'cluster_idx': 'source_cluster'}
            ),
            on='source',
            how='left'
        )
        # Add target cluster info
        self.edges = self.edges.merge(
            self.nodes[['node_id', 'date', 'cluster_idx']].rename(
                columns={'node_id': 'target', 'date': 'target_date', 'cluster_idx': 'target_cluster'}
            ),
            on='target',
            how='left'
        )
        
        # Extract cluster membership from nodes
        # Each node has a list of tickers in the 'elements' column
        records = []
        for _, node in self.nodes.iterrows():
            tickers = [t.strip() for t in node['elements'].split(',')]
            for ticker in tickers:
                records.append({
                    'date': node['date'],
                    'cluster_idx': node['cluster_idx'],
                    'ticker': ticker
                })
        self.cluster_membership = pd.DataFrame(records)
        
        print(f"  ✓ Loaded {len(self.edges)} edges")
        print(f"  ✓ Loaded {len(self.nodes)} nodes")
        print(f"  ✓ Loaded {len(self.cluster_membership)} cluster memberships")
        
        # Precompute metrics
        self._compute_cluster_metrics()
    
    def _compute_cluster_metrics(self):
        """Precompute metrics for each cluster on each day."""
        print("Computing cluster metrics...")
        
        # For each cluster, compute:
        # 1. Average incoming edge weight (persistence from previous day)
        # 2. Average outgoing edge weight (persistence to next day)
        # 3. Number of stocks
        # 4. Is it a new cluster? (no incoming edges)
        # 5. Is it dying? (no outgoing edges)
        
        metrics = []
        
        for _, node in self.nodes.iterrows():
            date = node['date']
            cluster_idx = node['cluster_idx']
            size = node['cluster_size']
            
            # Incoming edges (from previous day)
            incoming = self.edges[
                (self.edges['target_date'] == date) & 
                (self.edges['target_cluster'] == cluster_idx)
            ]
            
            # Outgoing edges (to next day)
            outgoing = self.edges[
                (self.edges['source_date'] == date) & 
                (self.edges['source_cluster'] == cluster_idx)
            ]
            
            avg_incoming_weight = incoming['intersection_weight'].mean() if len(incoming) > 0 else 0
            avg_outgoing_weight = outgoing['intersection_weight'].mean() if len(outgoing) > 0 else 0
            
            is_new = len(incoming) == 0  # No incoming edges = new cluster
            is_dying = len(outgoing) == 0  # No outgoing edges = dying cluster
            
            # Persistence score: how well this cluster persists
            persistence = (avg_incoming_weight + avg_outgoing_weight) / 2 if size > 0 else 0
            
            metrics.append({
                'date': date,
                'cluster_idx': cluster_idx,
                'size': size,
                'avg_incoming_weight': avg_incoming_weight,
                'avg_outgoing_weight': avg_outgoing_weight,
                'persistence_score': persistence,
                'is_new': is_new,
                'is_dying': is_dying,
                'num_incoming_edges': len(incoming),
                'num_outgoing_edges': len(outgoing)
            })
        
        self.cluster_metrics = pd.DataFrame(metrics)
        print(f"  ✓ Computed metrics for {len(self.cluster_metrics)} cluster-days")
    
    def generate_signals(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Generate trading signals for a specific date based on TGC dynamics.
        
        Signals range from -1 (strong sell) to +1 (strong buy).
        
        Strategy logic:
        1. PERSISTENT THEMES (+1.0): Buy stocks in large, persistent clusters
        2. EMERGING THEMES (+0.75): Buy stocks in new clusters that are growing
        3. DYING THEMES (-1.0): Sell stocks in clusters with declining persistence
        4. STRENGTHENING THEMES (+0.5): Buy stocks in clusters with increasing edge weights
        5. WEAKENING THEMES (-0.5): Sell stocks in clusters with decreasing edge weights
        """
        signals = {}
        
        # Get today's clusters
        today_metrics = self.cluster_metrics[self.cluster_metrics['date'] == date]
        
        if len(today_metrics) == 0:
            return signals
        
        # Get cluster membership for today
        today_membership = self.cluster_membership[self.cluster_membership['date'] == date]
        
        for _, cluster in today_metrics.iterrows():
            cluster_idx = cluster['cluster_idx']
            size = cluster['size']
            persistence = cluster['persistence_score']
            is_new = cluster['is_new']
            is_dying = cluster['is_dying']
            avg_incoming = cluster['avg_incoming_weight']
            avg_outgoing = cluster['avg_outgoing_weight']
            
            # Get stocks in this cluster
            stocks = today_membership[
                today_membership['cluster_idx'] == cluster_idx
            ]['ticker'].tolist()
            
            # Signal 1: PERSISTENT THEMES
            # Large clusters (>10 stocks) with high persistence (>20 edge weight)
            if size > 10 and persistence > 20:
                signal_strength = min(persistence / 50, 1.0)  # Cap at 1.0
                for stock in stocks:
                    signals[stock] = signals.get(stock, 0) + signal_strength
            
            # Signal 2: EMERGING THEMES
            # New clusters that are large (potential new theme forming)
            elif is_new and size > 5:
                signal_strength = min(size / 20, 0.75)  # Cap at 0.75
                for stock in stocks:
                    signals[stock] = signals.get(stock, 0) + signal_strength
            
            # Signal 3: DYING THEMES
            # Clusters with no outgoing edges (theme dissolving)
            elif is_dying and not is_new:  # Ignore if both new and dying (noise)
                for stock in stocks:
                    signals[stock] = signals.get(stock, 0) - 1.0
            
            # Signal 4: STRENGTHENING THEMES
            # Clusters where outgoing > incoming (growing persistence)
            elif avg_outgoing > avg_incoming * 1.2 and size > 3:
                growth_rate = (avg_outgoing - avg_incoming) / (avg_incoming + 1)
                signal_strength = min(growth_rate, 0.5)
                for stock in stocks:
                    signals[stock] = signals.get(stock, 0) + signal_strength
            
            # Signal 5: WEAKENING THEMES
            # Clusters where incoming > outgoing (declining persistence)
            elif avg_incoming > avg_outgoing * 1.2 and size > 3:
                decline_rate = (avg_incoming - avg_outgoing) / (avg_incoming + 1)
                signal_strength = min(decline_rate, 0.5)
                for stock in stocks:
                    signals[stock] = signals.get(stock, 0) - signal_strength
        
        # Normalize signals to [-1, 1] range
        if signals:
            max_abs_signal = max(abs(s) for s in signals.values())
            if max_abs_signal > 1:
                signals = {k: v / max_abs_signal for k, v in signals.items()}
        
        return signals
    
    def get_signal_explanation(self, date: pd.Timestamp, ticker: str) -> str:
        """Get human-readable explanation of why a signal was generated."""
        # Find which cluster(s) this ticker belongs to
        clusters = self.cluster_membership[
            (self.cluster_membership['date'] == date) & 
            (self.cluster_membership['ticker'] == ticker)
        ]['cluster_idx'].tolist()
        
        explanations = []
        
        for cluster_idx in clusters:
            metrics = self.cluster_metrics[
                (self.cluster_metrics['date'] == date) & 
                (self.cluster_metrics['cluster_idx'] == cluster_idx)
            ]
            
            if len(metrics) == 0:
                continue
            
            m = metrics.iloc[0]
            
            if m['size'] > 10 and m['persistence_score'] > 20:
                explanations.append(
                    f"PERSISTENT: In large cluster ({m['size']} stocks) with "
                    f"high persistence ({m['persistence_score']:.1f})"
                )
            
            if m['is_new'] and m['size'] > 5:
                explanations.append(
                    f"EMERGING: New cluster forming with {m['size']} stocks"
                )
            
            if m['is_dying'] and not m['is_new']:
                explanations.append(
                    f"DYING: Cluster dissolving (no outgoing edges)"
                )
            
            if m['avg_outgoing_weight'] > m['avg_incoming_weight'] * 1.2:
                explanations.append(
                    f"STRENGTHENING: Edge weight growing "
                    f"({m['avg_incoming_weight']:.1f} → {m['avg_outgoing_weight']:.1f})"
                )
            
            if m['avg_incoming_weight'] > m['avg_outgoing_weight'] * 1.2:
                explanations.append(
                    f"WEAKENING: Edge weight declining "
                    f"({m['avg_incoming_weight']:.1f} → {m['avg_outgoing_weight']:.1f})"
                )
        
        return " | ".join(explanations) if explanations else "No strong signal"
    
    def backtest(self, 
                 price_data: pd.DataFrame,
                 initial_capital: float = 100000,
                 signal_threshold: float = 0.3,
                 max_positions: int = 20) -> Dict:
        """
        Simple backtest of temporal graph strategy.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Index: dates, Columns: tickers
        initial_capital : float
            Starting capital
        signal_threshold : float
            Minimum signal strength to trade
        max_positions : int
            Maximum concurrent positions
        """
        from .trading_strategy import SimpleBacktester
        
        backtester = SimpleBacktester(
            initial_capital=initial_capital,
            commission=0.001,
            max_positions=max_positions
        )
        
        dates = sorted(self.cluster_metrics['date'].unique())
        
        print(f"\nBacktesting temporal graph strategy...")
        print(f"  Signal threshold: {signal_threshold}")
        print(f"  Max positions: {max_positions}")
        print(f"  Trading days: {len(dates)}\n")
        
        matched_dates = 0
        for i, date in enumerate(dates):
            if date not in price_data.index:
                continue
            
            matched_dates += 1
            
            # Generate signals
            signals_dict = self.generate_signals(date)
            
            # Convert to DataFrame format
            if len(signals_dict) > 0:
                signals_df = pd.DataFrame([
                    {'ticker': k, 'signal': v, 'momentum': 0, 'mean_reversion': 0, 'rotation': 0}
                    for k, v in signals_dict.items()
                ]).sort_values('signal', ascending=False)
            else:
                signals_df = pd.DataFrame(columns=['ticker', 'signal', 'momentum', 'mean_reversion', 'rotation'])
            
            # Execute trades
            prices = price_data.loc[date]
            backtester.execute_signals(
                date=date,
                signals=signals_df,
                prices=prices,
                signal_threshold=signal_threshold
            )
            
            if (i + 1) % 50 == 0:
                print(f"  - Processed {i + 1}/{len(dates)} days...")
        
        print(f"\n  ✓ Matched {matched_dates}/{len(dates)} dates with price data")
        
        return backtester.get_performance_stats()


if __name__ == "__main__":
    print("Temporal Graph Cluster Trading Strategy")
    print("=" * 70)
    print("\nThis strategy trades based on temporal cluster dynamics:")
    print("  • PERSISTENT themes (high edge weights)")
    print("  • EMERGING themes (new clusters forming)")
    print("  • DYING themes (clusters dissolving)")
    print("  • STRENGTHENING themes (increasing edge weights)")
    print("  • WEAKENING themes (decreasing edge weights)")
    print("\nSee example_temporal_backtest.py for usage.")

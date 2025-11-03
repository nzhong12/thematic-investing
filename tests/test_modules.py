"""Tests for PGI Theme Graphs modules."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestCorrelation(unittest.TestCase):
    """Test correlation module."""

    def setUp(self):
        """Create sample returns data."""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        np.random.seed(42)
        
        # Create 5 stocks with some correlation structure
        n_stocks = 5
        data = np.random.randn(60, n_stocks).cumsum(axis=0)
        
        self.returns_df = pd.DataFrame(
            data,
            index=dates,
            columns=[f'TICK{i}' for i in range(n_stocks)]
        )

    def test_rolling_correlation_initialization(self):
        """Test RollingCorrelation initialization."""
        from src.pgi_theme_graphs.correlation import RollingCorrelation
        
        corr_analyzer = RollingCorrelation(self.returns_df)
        self.assertIsNotNone(corr_analyzer)
        self.assertTrue(corr_analyzer.returns_df.equals(self.returns_df))

    def test_compute_rolling_correlations(self):
        """Test rolling correlation computation."""
        from src.pgi_theme_graphs.correlation import RollingCorrelation
        
        corr_analyzer = RollingCorrelation(self.returns_df)
        windows = [10, 20]
        results = corr_analyzer.compute_rolling_correlations(windows=windows)
        
        # Check that both windows are computed
        self.assertEqual(len(results), 2)
        self.assertIn(10, results)
        self.assertIn(20, results)
        
        # Check data format
        for window, corr_df in results.items():
            self.assertIsInstance(corr_df, pd.DataFrame)
            self.assertIn('date', corr_df.columns)
            self.assertIn('ticker1', corr_df.columns)
            self.assertIn('ticker2', corr_df.columns)
            self.assertIn('correlation', corr_df.columns)


class TestGraphBuilder(unittest.TestCase):
    """Test graph builder module."""

    def setUp(self):
        """Create sample distance matrix."""
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # Create symmetric distance matrix
        distances = np.array([
            [0.0, 0.5, 0.4, 0.7, 0.9],
            [0.5, 0.0, 0.3, 0.6, 0.8],
            [0.4, 0.3, 0.0, 0.5, 0.7],
            [0.7, 0.6, 0.5, 0.0, 0.4],
            [0.9, 0.8, 0.7, 0.4, 0.0]
        ])
        
        self.distance_matrix = pd.DataFrame(
            distances,
            index=tickers,
            columns=tickers
        )

    def test_graph_initialization(self):
        """Test ThemeGraph initialization."""
        from src.pgi_theme_graphs.graph_builder import ThemeGraph
        
        graph = ThemeGraph(self.distance_matrix)
        self.assertIsNotNone(graph)
        self.assertTrue(graph.distance_matrix.equals(self.distance_matrix))

    def test_build_complete_graph(self):
        """Test complete graph construction."""
        from src.pgi_theme_graphs.graph_builder import ThemeGraph
        
        theme_graph = ThemeGraph(self.distance_matrix)
        G = theme_graph.build_complete_graph()
        
        # Check graph properties
        self.assertEqual(G.number_of_nodes(), 5)
        self.assertGreater(G.number_of_edges(), 0)
        
        # Check that edges have weights
        for _, _, data in G.edges(data=True):
            self.assertIn('weight', data)
            self.assertGreater(data['weight'], 0)

    def test_build_mst(self):
        """Test MST construction."""
        from src.pgi_theme_graphs.graph_builder import ThemeGraph
        
        theme_graph = ThemeGraph(self.distance_matrix)
        mst = theme_graph.build_mst()
        
        # MST should have n-1 edges for n nodes
        self.assertEqual(mst.number_of_nodes(), 5)
        self.assertEqual(mst.number_of_edges(), 4)
        
        # Check connectivity
        import networkx as nx
        self.assertTrue(nx.is_connected(mst))


class TestVisualization(unittest.TestCase):
    """Test visualization module."""

    def test_visualizer_initialization(self):
        """Test ThemeVisualizer initialization."""
        from src.pgi_theme_graphs.visualization import ThemeVisualizer
        
        viz = ThemeVisualizer(figsize=(10, 8))
        self.assertIsNotNone(viz)
        self.assertEqual(viz.figsize, (10, 8))


if __name__ == '__main__':
    unittest.main()

"""Main pipeline for PGI Theme Graphs analysis."""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .data_loader import CRSPDataLoader
from .correlation import RollingCorrelation
from .graph_builder import ThemeGraph
from .visualization import ThemeVisualizer


class ThemeGraphPipeline:
    """End-to-end pipeline for market theme analysis."""

    def __init__(
        self,
        wrds_username: Optional[str] = None,
        windows: List[int] = [10, 30, 50]
    ):
        """Initialize pipeline.
        
        Args:
            wrds_username: WRDS username for database access.
            windows: List of rolling window sizes in days.
        """
        self.data_loader = CRSPDataLoader(username=wrds_username)
        self.windows = windows
        self.returns_df = None
        self.correlation_analyzer = None
        self.visualizer = ThemeVisualizer()
        
    def load_data(
        self,
        start_date: str,
        end_date: str,
        permnos: Optional[list] = None
    ) -> pd.DataFrame:
        """Load S&P 500 returns data from WRDS.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            permnos: Optional list of PERMNOs to filter.
            
        Returns:
            DataFrame with returns in pivot format.
        """
        print(f"Loading data from {start_date} to {end_date}...")
        
        with self.data_loader as loader:
            self.returns_df = loader.get_returns_pivot(
                start_date=start_date,
                end_date=end_date,
                permnos=permnos
            )
        
        print(f"Loaded {len(self.returns_df)} dates and {len(self.returns_df.columns)} tickers")
        return self.returns_df
    
    def compute_correlations(
        self,
        min_periods: Optional[int] = None
    ) -> Dict[int, pd.DataFrame]:
        """Compute rolling correlation matrices.
        
        Args:
            min_periods: Minimum observations required for correlation.
            
        Returns:
            Dictionary mapping window size to correlation data.
        """
        if self.returns_df is None:
            raise ValueError("Must load data first using load_data()")
        
        print(f"Computing rolling correlations for windows: {self.windows}")
        
        self.correlation_analyzer = RollingCorrelation(self.returns_df)
        correlations = self.correlation_analyzer.compute_rolling_correlations(
            windows=self.windows,
            min_periods=min_periods
        )
        
        print("Correlation computation complete")
        return correlations
    
    def analyze_date(
        self,
        date: str,
        window: int,
        distance_method: str = 'angular',
        community_method: str = 'louvain'
    ) -> Dict:
        """Analyze market themes for a specific date and window.
        
        Args:
            date: Date string in 'YYYY-MM-DD' format.
            window: Rolling window size.
            distance_method: Distance metric for graph construction.
            community_method: Community detection algorithm.
            
        Returns:
            Dictionary containing analysis results.
        """
        if self.correlation_analyzer is None:
            raise ValueError("Must compute correlations first")
        
        # Convert to timestamp
        date_ts = pd.Timestamp(date)
        
        # Get distance matrix
        dist_matrix = self.correlation_analyzer.get_distance_matrix(
            window=window,
            date=date_ts,
            method=distance_method
        )
        
        # Build graphs
        theme_graph = ThemeGraph(dist_matrix)
        complete_graph = theme_graph.build_complete_graph()
        mst = theme_graph.build_mst()
        
        # Detect communities
        communities = theme_graph.detect_communities(
            method=community_method,
            graph_type='mst'
        )
        
        # Compute metrics
        properties = theme_graph.get_graph_properties()
        centrality = theme_graph.get_centrality_measures(graph_type='mst')
        
        return {
            'date': date,
            'window': window,
            'distance_matrix': dist_matrix,
            'complete_graph': complete_graph,
            'mst': mst,
            'communities': communities,
            'properties': properties,
            'centrality': centrality
        }
    
    def analyze_evolution(
        self,
        window: int,
        date_range: Optional[Tuple[str, str]] = None,
        sample_freq: str = 'W',
        distance_method: str = 'angular',
        community_method: str = 'louvain'
    ) -> List[Dict]:
        """Analyze evolution of themes over time.
        
        Args:
            window: Rolling window size to analyze.
            date_range: Optional tuple of (start_date, end_date).
            sample_freq: Sampling frequency ('D', 'W', 'M').
            distance_method: Distance metric for graph construction.
            community_method: Community detection algorithm.
            
        Returns:
            List of analysis results for each sampled date.
        """
        if self.correlation_analyzer is None:
            raise ValueError("Must compute correlations first")
        
        # Get available dates
        all_dates = self.correlation_analyzer.get_all_dates(window)
        
        # Filter date range if provided
        if date_range:
            start, end = date_range
            all_dates = [d for d in all_dates if start <= str(d) <= end]
        
        # Sample dates
        dates_df = pd.DataFrame({'date': all_dates})
        dates_df.set_index('date', inplace=True)
        sampled_dates = dates_df.resample(sample_freq).first().index.tolist()
        
        print(f"Analyzing evolution across {len(sampled_dates)} dates...")
        
        # Analyze each date
        results = []
        for i, date in enumerate(sampled_dates, 1):
            print(f"Processing {i}/{len(sampled_dates)}: {date}")
            result = self.analyze_date(
                date=str(date.date()),
                window=window,
                distance_method=distance_method,
                community_method=community_method
            )
            results.append(result)
        
        return results
    
    def visualize_results(
        self,
        results: List[Dict],
        output_dir: str = './output'
    ) -> None:
        """Generate visualizations for analysis results.
        
        Args:
            results: List of analysis results from analyze_evolution().
            output_dir: Directory to save visualizations.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating visualizations in {output_dir}...")
        
        # Plot MST for first, middle, and last dates
        sample_indices = [0, len(results) // 2, -1]
        
        for idx in sample_indices:
            result = results[idx]
            date_str = result['date'].replace('-', '')
            window = result['window']
            
            # MST visualization
            self.visualizer.plot_mst(
                mst=result['mst'],
                communities=result['communities'],
                title=f"MST - {result['date']} ({window}-day window)",
                save_path=f"{output_dir}/mst_{date_str}_w{window}.png"
            )
            
        # Community evolution
        community_timeline = [r['communities'] for r in results]
        dates = [r['date'] for r in results]
        
        self.visualizer.plot_community_evolution(
            community_timeline=community_timeline,
            dates=dates,
            title=f"Theme Evolution ({results[0]['window']}-day window)",
            save_path=f"{output_dir}/community_evolution_w{results[0]['window']}.png"
        )
        
        print(f"Visualizations saved to {output_dir}")

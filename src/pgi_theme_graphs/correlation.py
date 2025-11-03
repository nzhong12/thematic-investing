"""Correlation module for computing rolling correlation matrices."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class RollingCorrelation:
    """Compute rolling correlation matrices for stock returns."""

    def __init__(self, returns_df: pd.DataFrame):
        """Initialize with returns data.
        
        Args:
            returns_df: DataFrame with dates as index and tickers as columns.
        """
        self.returns_df = returns_df
        self.correlation_matrices = {}

    def compute_rolling_correlations(
        self,
        windows: List[int] = [10, 30, 50],
        min_periods: int = None
    ) -> Dict[int, pd.DataFrame]:
        """Compute rolling correlation matrices for multiple windows.
        
        Args:
            windows: List of window sizes (e.g., [10, 30, 50] days).
            min_periods: Minimum observations required. Defaults to window size.
            
        Returns:
            Dictionary mapping window size to 3D correlation data.
            Each value is a DataFrame with MultiIndex (date, ticker1, ticker2).
        """
        for window in windows:
            if min_periods is None:
                min_obs = window
            else:
                min_obs = min_periods

            # Compute rolling correlation for each window
            rolling_corr = self._compute_single_window(window, min_obs)
            self.correlation_matrices[window] = rolling_corr

        return self.correlation_matrices

    def _compute_single_window(
        self,
        window: int,
        min_periods: int
    ) -> pd.DataFrame:
        """Compute rolling correlation for a single window size.
        
        Args:
            window: Window size in days.
            min_periods: Minimum observations required.
            
        Returns:
            DataFrame with MultiIndex (date, ticker1, ticker2) and correlation values.
        """
        # Rolling window correlation
        rolling_obj = self.returns_df.rolling(window=window, min_periods=min_periods)
        
        # Store correlation matrices for each date
        corr_data = []
        
        for date in self.returns_df.index[window-1:]:
            # Get window of data
            window_data = self.returns_df.loc[:date].tail(window)
            
            # Compute correlation matrix
            if len(window_data) >= min_periods:
                corr_matrix = window_data.corr()
                
                # Convert to long format
                for ticker1 in corr_matrix.index:
                    for ticker2 in corr_matrix.columns:
                        corr_data.append({
                            'date': date,
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'correlation': corr_matrix.loc[ticker1, ticker2]
                        })
        
        return pd.DataFrame(corr_data)

    def get_correlation_matrix(
        self,
        window: int,
        date: pd.Timestamp
    ) -> pd.DataFrame:
        """Get correlation matrix for a specific window and date.
        
        Args:
            window: Window size in days.
            date: Date for which to get correlation matrix.
            
        Returns:
            Correlation matrix as DataFrame.
        """
        if window not in self.correlation_matrices:
            raise ValueError(f"Window {window} not computed. Available: {list(self.correlation_matrices.keys())}")
        
        # Filter for specific date
        corr_df = self.correlation_matrices[window]
        date_data = corr_df[corr_df['date'] == date]
        
        # Pivot to matrix format
        matrix = date_data.pivot(index='ticker1', columns='ticker2', values='correlation')
        
        return matrix

    def get_distance_matrix(
        self,
        window: int,
        date: pd.Timestamp,
        method: str = 'angular'
    ) -> pd.DataFrame:
        """Convert correlation to distance matrix.
        
        Args:
            window: Window size in days.
            date: Date for which to get distance matrix.
            method: Distance metric ('angular', 'euclidean').
            
        Returns:
            Distance matrix as DataFrame.
        """
        corr_matrix = self.get_correlation_matrix(window, date)
        
        if method == 'angular':
            # Angular distance: sqrt(2 * (1 - correlation))
            dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        elif method == 'euclidean':
            # Euclidean distance: sqrt(2 - 2*correlation)
            dist_matrix = np.sqrt(2 - 2 * corr_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return dist_matrix

    def get_all_dates(self, window: int) -> List[pd.Timestamp]:
        """Get all dates for which correlations are available.
        
        Args:
            window: Window size in days.
            
        Returns:
            List of dates.
        """
        if window not in self.correlation_matrices:
            raise ValueError(f"Window {window} not computed.")
        
        return sorted(self.correlation_matrices[window]['date'].unique())

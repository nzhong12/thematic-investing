"""Visualization module for market theme clusters and network graphs."""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from matplotlib.animation import FuncAnimation


class ThemeVisualizer:
    """Visualize evolving market theme clusters using network graphs."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.
        
        Args:
            figsize: Figure size for plots.
        """
        self.figsize = figsize
        sns.set_style("whitegrid")

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
        cmap: str = "coolwarm",
        vmin: float = -1,
        vmax: float = 1,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot correlation matrix as heatmap.
        
        Args:
            corr_matrix: Correlation matrix DataFrame.
            title: Plot title.
            cmap: Colormap name.
            vmin: Minimum color value.
            vmax: Maximum color value.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_network_graph(
        self,
        graph: nx.Graph,
        communities: Optional[Dict[str, int]] = None,
        title: str = "Market Theme Network",
        layout: str = 'spring',
        node_size: int = 500,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot network graph with optional community coloring.
        
        Args:
            graph: NetworkX graph.
            communities: Dictionary mapping nodes to community IDs.
            title: Plot title.
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai').
            node_size: Size of nodes.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, weight='weight', seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph, weight='weight')
        else:
            pos = nx.spring_layout(graph)
        
        # Determine node colors
        if communities:
            node_colors = [communities.get(node, 0) for node in graph.nodes()]
            cmap = plt.cm.tab20
        else:
            node_colors = 'lightblue'
            cmap = None
        
        # Draw network
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            cmap=cmap,
            alpha=0.8,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            graph,
            pos,
            width=1.0,
            alpha=0.5,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_mst(
        self,
        mst: nx.Graph,
        communities: Optional[Dict[str, int]] = None,
        title: str = "Minimum Spanning Tree",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot minimum spanning tree.
        
        Args:
            mst: Minimum spanning tree NetworkX graph.
            communities: Dictionary mapping nodes to community IDs.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        return self.plot_network_graph(
            mst,
            communities=communities,
            title=title,
            layout='kamada_kawai',
            node_size=700,
            save_path=save_path
        )

    def plot_community_evolution(
        self,
        community_timeline: List[Dict[str, int]],
        dates: List[str],
        title: str = "Community Evolution Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot evolution of communities over time.
        
        Args:
            community_timeline: List of community dictionaries for each date.
            dates: List of date strings corresponding to each community dict.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        # Convert to DataFrame for easier plotting
        data = []
        for date, communities in zip(dates, community_timeline):
            for ticker, comm_id in communities.items():
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'community': comm_id
                })
        
        df = pd.DataFrame(data)
        
        # Create pivot for heatmap
        pivot = df.pivot(index='ticker', columns='date', values='community')
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(8, len(pivot) * 0.3)))
        
        sns.heatmap(
            pivot,
            cmap='tab20',
            cbar_kws={"label": "Community ID"},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Ticker", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_centrality_distribution(
        self,
        centrality: Dict[str, float],
        title: str = "Node Centrality Distribution",
        xlabel: str = "Centrality Score",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution of centrality scores.
        
        Args:
            centrality: Dictionary mapping nodes to centrality scores.
            title: Plot title.
            xlabel: X-axis label.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        scores = list(centrality.values())
        
        ax.hist(scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_rolling_metrics(
        self,
        metrics_df: pd.DataFrame,
        metric_name: str,
        windows: List[int] = [10, 30, 50],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot rolling metrics over time for different windows.
        
        Args:
            metrics_df: DataFrame with date index and metric columns.
            metric_name: Name of metric to plot.
            windows: List of window sizes to display.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for window in windows:
            col_name = f"{metric_name}_{window}"
            if col_name in metrics_df.columns:
                ax.plot(
                    metrics_df.index,
                    metrics_df[col_name],
                    label=f"{window}-day window",
                    linewidth=2
                )
        
        if title is None:
            title = f"Rolling {metric_name} Over Time"
        
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

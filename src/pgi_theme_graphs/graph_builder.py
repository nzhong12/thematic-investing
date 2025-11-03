"""Graph construction module for distance-weighted graphs and MST."""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.sparse.csgraph import minimum_spanning_tree


class ThemeGraph:
    """Construct distance-weighted graphs and minimum spanning trees."""

    def __init__(self, distance_matrix: pd.DataFrame):
        """Initialize with distance matrix.
        
        Args:
            distance_matrix: DataFrame of pairwise distances between tickers.
        """
        self.distance_matrix = distance_matrix
        self.graph = None
        self.mst = None

    def build_complete_graph(self, threshold: Optional[float] = None) -> nx.Graph:
        """Build complete weighted graph from distance matrix.
        
        Args:
            threshold: Optional distance threshold to filter edges.
            
        Returns:
            NetworkX graph with edge weights as distances.
        """
        G = nx.Graph()
        
        # Add nodes
        tickers = self.distance_matrix.index.tolist()
        G.add_nodes_from(tickers)
        
        # Add edges with weights
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i < j:  # Avoid duplicates
                    distance = self.distance_matrix.loc[ticker1, ticker2]
                    
                    # Skip if NaN or above threshold
                    if pd.notna(distance):
                        if threshold is None or distance <= threshold:
                            G.add_edge(ticker1, ticker2, weight=distance)
        
        self.graph = G
        return G

    def build_mst(self) -> nx.Graph:
        """Build minimum spanning tree using Kruskal's algorithm.
        
        Returns:
            NetworkX graph representing the MST.
        """
        if self.graph is None:
            self.build_complete_graph()
        
        # Use NetworkX MST algorithm
        mst = nx.minimum_spanning_tree(self.graph, weight='weight')
        self.mst = mst
        
        return mst

    def build_mst_scipy(self) -> nx.Graph:
        """Build MST using scipy's efficient sparse implementation.
        
        Returns:
            NetworkX graph representing the MST.
        """
        # Convert distance matrix to numpy array
        dist_array = self.distance_matrix.values
        
        # Handle NaN values
        dist_array = np.nan_to_num(dist_array, nan=np.inf)
        
        # Compute MST using scipy
        mst_scipy = minimum_spanning_tree(dist_array)
        
        # Convert to NetworkX graph
        mst = nx.Graph()
        tickers = self.distance_matrix.index.tolist()
        mst.add_nodes_from(tickers)
        
        # Add edges from sparse matrix
        rows, cols = mst_scipy.nonzero()
        for i, j in zip(rows, cols):
            if i < j:
                weight = mst_scipy[i, j]
                mst.add_edge(tickers[i], tickers[j], weight=weight)
        
        self.mst = mst
        return mst

    def detect_communities(
        self,
        method: str = 'louvain',
        graph_type: str = 'mst'
    ) -> Dict[str, int]:
        """Detect market theme communities/clusters.
        
        Args:
            method: Community detection method ('louvain', 'girvan_newman').
            graph_type: Which graph to use ('mst' or 'complete').
            
        Returns:
            Dictionary mapping ticker to community ID.
        """
        # Select graph
        if graph_type == 'mst':
            if self.mst is None:
                self.build_mst()
            G = self.mst
        else:
            if self.graph is None:
                self.build_complete_graph()
            G = self.graph

        # Apply community detection
        if method == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G)
            except ImportError:
                # Fallback to greedy modularity
                communities_gen = nx.community.greedy_modularity_communities(G)
                communities = {}
                for idx, comm in enumerate(communities_gen):
                    for node in comm:
                        communities[node] = idx
        
        elif method == 'girvan_newman':
            # Use first level of Girvan-Newman
            communities_gen = nx.community.girvan_newman(G)
            first_level = next(communities_gen)
            communities = {}
            for idx, comm in enumerate(first_level):
                for node in comm:
                    communities[node] = idx
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return communities

    def get_graph_properties(self) -> Dict:
        """Compute graph properties and metrics.
        
        Returns:
            Dictionary of graph properties.
        """
        if self.graph is None:
            self.build_complete_graph()
        
        properties = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
        }
        
        # Add MST properties if available
        if self.mst is not None:
            properties['mst_edges'] = self.mst.number_of_edges()
            properties['mst_total_weight'] = sum(
                data['weight'] for _, _, data in self.mst.edges(data=True)
            )
        
        return properties

    def get_centrality_measures(
        self,
        graph_type: str = 'mst'
    ) -> Dict[str, Dict[str, float]]:
        """Compute various centrality measures.
        
        Args:
            graph_type: Which graph to use ('mst' or 'complete').
            
        Returns:
            Dictionary of centrality measures for each node.
        """
        # Select graph
        if graph_type == 'mst':
            if self.mst is None:
                self.build_mst()
            G = self.mst
        else:
            if self.graph is None:
                self.build_complete_graph()
            G = self.graph

        centrality = {
            'degree': dict(G.degree()),
            'betweenness': nx.betweenness_centrality(G, weight='weight'),
            'closeness': nx.closeness_centrality(G, distance='weight'),
        }
        
        return centrality

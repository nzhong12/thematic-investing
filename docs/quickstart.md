# PGI Theme Graphs - Quick Start Guide

## What This Tool Does

Analyzes evolving market themes in the S&P 500 by:
1. Pulling daily stock returns from WRDS CRSP database
2. Computing rolling correlations between stocks
3. Building network graphs based on correlation distances
4. Identifying market theme clusters using community detection
5. Visualizing how these themes evolve over time

## Installation

```bash
pip install -r requirements.txt
```

## WRDS Setup

Set your WRDS username as an environment variable:
```bash
export WRDS_USERNAME='your_username'
```

Or create a `.pgpass` file for automatic authentication.

## Basic Usage

```python
from src.pgi_theme_graphs.pipeline import ThemeGraphPipeline

# Initialize
pipeline = ThemeGraphPipeline(windows=[10, 30, 50])

# Load data
pipeline.load_data(
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Compute correlations
pipeline.compute_correlations()

# Analyze a specific date
result = pipeline.analyze_date(
    date='2023-06-30',
    window=30
)

# Access results
print(f"Communities: {result['communities']}")
print(f"Graph properties: {result['properties']}")

# Analyze evolution over time
results = pipeline.analyze_evolution(
    window=30,
    sample_freq='W'  # Weekly sampling
)

# Generate visualizations
pipeline.visualize_results(results, output_dir='./output')
```

## Key Components

### Data Loader
```python
from src.pgi_theme_graphs.data_loader import CRSPDataLoader

with CRSPDataLoader() as loader:
    returns = loader.get_returns_pivot(
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
```

### Correlation Analysis
```python
from src.pgi_theme_graphs.correlation import RollingCorrelation

corr_analyzer = RollingCorrelation(returns_df)
correlations = corr_analyzer.compute_rolling_correlations([10, 30, 50])

# Get distance matrix for specific date
dist_matrix = corr_analyzer.get_distance_matrix(
    window=30,
    date=pd.Timestamp('2023-06-30')
)
```

### Graph Construction
```python
from src.pgi_theme_graphs.graph_builder import ThemeGraph

graph = ThemeGraph(distance_matrix)
mst = graph.build_mst()
communities = graph.detect_communities(method='louvain')
centrality = graph.get_centrality_measures()
```

### Visualization
```python
from src.pgi_theme_graphs.visualization import ThemeVisualizer

viz = ThemeVisualizer()

# Plot MST with communities
viz.plot_mst(
    mst=mst,
    communities=communities,
    save_path='mst.png'
)

# Plot community evolution
viz.plot_community_evolution(
    community_timeline=communities_list,
    dates=dates_list,
    save_path='evolution.png'
)
```

## Common Parameters

### Rolling Windows
- **10 days**: Short-term correlations, sensitive to recent moves
- **30 days**: Medium-term, captures monthly themes
- **50 days**: Longer-term, more stable patterns

### Distance Metrics
- **Angular**: `sqrt(2 * (1 - correlation))` - emphasizes negative correlations
- **Euclidean**: `sqrt(2 - 2*correlation)` - standard metric

### Community Detection
- **Louvain**: Fast, modularity-based, good for large graphs
- **Girvan-Newman**: Hierarchical, slower but more interpretable

### Sampling Frequencies
- **'D'**: Daily - high resolution, many data points
- **'W'**: Weekly - good balance for multi-year analyses
- **'M'**: Monthly - overview of long-term trends

## Example Output

### Graph Properties
```python
{
    'num_nodes': 503,
    'num_edges': 126253,
    'density': 0.998,
    'avg_clustering': 0.996,
    'mst_edges': 502,
    'mst_total_weight': 387.4
}
```

### Centrality Measures
Most central stocks indicate "hub" companies that connect different market themes.

### Communities
Stocks grouped by similarity - might represent sectors, industries, or market themes.

## Troubleshooting

### WRDS Connection Issues
- Verify credentials: `~/.pgpass` or environment variable
- Check VPN if required by your institution
- Ensure CRSP subscription is active

### Memory Issues
- Reduce date range
- Use smaller stock universe (pass `permnos` list)
- Sample dates less frequently in evolution analysis

### Slow Performance
- Use weekly sampling instead of daily
- Compute one window at a time
- Consider scipy MST implementation for larger graphs

## File Structure

```
src/pgi_theme_graphs/
├── __init__.py          # Package initialization
├── data_loader.py       # WRDS data interface (147 lines)
├── correlation.py       # Rolling correlations (151 lines)
├── graph_builder.py     # Graph/MST construction (201 lines)
├── visualization.py     # Plotting functions (309 lines)
└── pipeline.py          # Orchestration (239 lines)
```

## Next Steps

1. **Explore your data**: Run on different time periods and windows
2. **Identify themes**: Look at which stocks cluster together
3. **Track evolution**: See how themes change during market events
4. **Develop strategies**: Use insights for the backtesting framework

## Resources

- **Full README**: Comprehensive project documentation
- **Architecture doc**: System design and technical details
- **Backtesting plan**: Future strategy implementation roadmap
- **Tests**: Unit tests demonstrating usage patterns

## Support

For issues or questions:
- Check documentation in `docs/`
- Review example usage in `src/example_usage.py`
- Run tests: `python -m unittest tests.test_modules`

# PGI Theme Graphs Project

A Python research tool for analyzing evolving market themes using network graph analysis of S&P 500 stock correlations.

## Overview

This project uses WRDS CRSP data to construct and visualize market theme clusters through:
- Rolling correlation analysis (10-, 30-, and 50-day windows)
- Distance-weighted network graphs
- Minimum spanning trees (MST)
- Community detection algorithms
- Interactive visualizations

## Features

### Data Acquisition
- Pull daily S&P 500 returns from WRDS CRSP database using SQL
- Efficient queries with date range filtering
- Support for custom stock universes

### Correlation Analysis
- Compute rolling correlation matrices for multiple time windows
- Convert correlations to distance metrics (angular, euclidean)
- Handle missing data and edge cases

### Graph Construction
- Build complete weighted graphs from distance matrices
- Construct minimum spanning trees using Kruskal's algorithm
- Detect market theme communities (Louvain, Girvan-Newman)
- Calculate centrality measures (betweenness, closeness, degree)

### Visualization
- Correlation heatmaps
- Network graph layouts (spring, circular, Kamada-Kawai)
- MST visualizations with community coloring
- Theme evolution timelines
- Centrality distributions

## Installation

```bash
# Clone the repository
git clone https://github.com/nzhong12/thematic-investing.git
cd thematic-investing

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- WRDS account with CRSP access
- See `requirements.txt` for package dependencies

## Usage

### Basic Example

```python
from src.pgi_theme_graphs.pipeline import ThemeGraphPipeline

# Initialize pipeline
pipeline = ThemeGraphPipeline(
    wrds_username='your_username',  # Or use environment variable
    windows=[10, 30, 50]
)

# Load S&P 500 returns
pipeline.load_data(
    start_date='2022-01-01',
    end_date='2022-12-31'
)

# Compute rolling correlations
pipeline.compute_correlations()

# Analyze theme evolution
results = pipeline.analyze_evolution(
    window=30,
    sample_freq='W',  # Weekly sampling
    distance_method='angular',
    community_method='louvain'
)

# Generate visualizations
pipeline.visualize_results(
    results=results,
    output_dir='./output/theme_graphs'
)
```

### Command Line

```bash
# Run example analysis
python src/example_usage.py
```

## Project Structure

```
thematic-investing/
├── src/
│   └── pgi_theme_graphs/
│       ├── __init__.py
│       ├── data_loader.py      # WRDS CRSP data interface
│       ├── correlation.py      # Rolling correlation computation
│       ├── graph_builder.py    # Graph construction & MST
│       ├── visualization.py    # Plotting functions
│       └── pipeline.py         # End-to-end workflow
├── docs/
│   ├── architecture.md         # System architecture overview
│   └── backtesting_plan.md     # Future backtesting implementation
├── tests/                      # Unit tests (to be added)
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## Methodology

### 1. Data Collection
Query daily returns for S&P 500 constituents from CRSP database:
```sql
SELECT a.date, a.permno, b.ticker, a.ret
FROM crsp.dsf as a
LEFT JOIN crsp.dsenames as b ON a.permno = b.permno
WHERE a.date BETWEEN 'start' AND 'end'
```

### 2. Correlation Computation
For each rolling window W (10, 30, 50 days):
- Compute Pearson correlation matrix ρ
- Convert to distance: d = sqrt(2 * (1 - ρ))

### 3. Graph Construction
- Build complete graph with distance-weighted edges
- Extract minimum spanning tree (MST)
- Apply community detection to identify market themes

### 4. Analysis
- Track community evolution over time
- Identify central stocks (high betweenness/closeness)
- Visualize theme clusters and transitions

## Configuration

### WRDS Credentials
Set environment variable:
```bash
export WRDS_USERNAME='your_username'
```

Or pass directly to pipeline:
```python
pipeline = ThemeGraphPipeline(wrds_username='your_username')
```

### Parameters
- **Rolling windows**: `[10, 30, 50]` days (customizable)
- **Distance metric**: `'angular'` or `'euclidean'`
- **Community detection**: `'louvain'` or `'girvan_newman'`
- **Sampling frequency**: `'D'` (daily), `'W'` (weekly), `'M'` (monthly)

## Future Work

### Backtesting Framework
Planned implementation of strategy backtesting:
- Community momentum strategies
- MST centrality-based portfolios
- Theme rotation algorithms
- Risk-adjusted performance metrics

See `docs/backtesting_plan.md` for detailed roadmap.

### Enhancements
- Real-time data streaming
- Interactive web dashboard
- Machine learning integration (GNNs)
- Cloud deployment

## Documentation

- **Architecture**: `docs/architecture.md` - System design and module descriptions
- **Backtesting Plan**: `docs/backtesting_plan.md` - Future strategy implementation

## Contributing

Contributions welcome! Please open issues or pull requests.

## License

MIT License

## Contact

For questions or collaboration: [Your Contact Info]

## Acknowledgments

- WRDS for CRSP database access
- NetworkX community for graph algorithms
- Research inspired by market network analysis literature

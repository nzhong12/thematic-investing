# PGI Theme Graphs - Architecture Overview

## System Architecture

### Data Flow
```
WRDS CRSP Database
       ↓
  Data Loader (SQL queries)
       ↓
Returns Matrix (dates × tickers)
       ↓
Correlation Module (rolling windows)
       ↓
Distance Matrices
       ↓
Graph Builder (MST, community detection)
       ↓
Visualization & Analysis
```

## Module Descriptions

### 1. `data_loader.py`
**Responsibility:** Interface with WRDS CRSP database

**Key Classes:**
- `CRSPDataLoader`: Fetches daily S&P 500 returns using SQL queries

**Dependencies:** wrds, pandas, sqlalchemy

**SQL Strategy:**
- Query `crsp.dsf` for daily stock file
- Join with `crsp.dsenames` for ticker symbols
- Filter S&P 500 constituents via `crsp.dsp500list`
- Return pivoted DataFrame (dates × tickers)

### 2. `correlation.py`
**Responsibility:** Compute rolling correlation matrices

**Key Classes:**
- `RollingCorrelation`: Manages correlation computation for multiple windows

**Methods:**
- Rolling window correlation (10, 30, 50 days)
- Distance matrix conversion (angular, euclidean)
- Efficient storage in long format

**Mathematical Foundation:**
- Pearson correlation coefficient
- Angular distance: `sqrt(2 * (1 - ρ))`
- Euclidean distance: `sqrt(2 - 2*ρ)`

### 3. `graph_builder.py`
**Responsibility:** Construct network graphs and detect communities

**Key Classes:**
- `ThemeGraph`: Builds weighted graphs and MSTs

**Algorithms:**
- Kruskal's MST (via NetworkX or scipy)
- Louvain community detection (modularity optimization)
- Girvan-Newman hierarchical clustering
- Centrality measures (betweenness, closeness, degree)

**Graph Types:**
- Complete graph: All pairwise distances
- MST: Minimum spanning tree (hierarchical structure)

### 4. `visualization.py`
**Responsibility:** Generate plots and visual analytics

**Key Classes:**
- `ThemeVisualizer`: Comprehensive visualization suite

**Visualizations:**
- Correlation heatmaps
- Network graph layouts (spring, circular, Kamada-Kawai)
- MST with community coloring
- Community evolution timelines
- Centrality distributions

### 5. `pipeline.py`
**Responsibility:** Orchestrate end-to-end analysis

**Key Classes:**
- `ThemeGraphPipeline`: High-level interface for complete workflow

**Workflow:**
1. Load data from WRDS
2. Compute rolling correlations
3. Analyze specific dates or time ranges
4. Detect communities
5. Generate visualizations

## Design Patterns

### Context Manager
- Data loader uses context manager for database connections
- Ensures proper cleanup of WRDS connections

### Builder Pattern
- Graph construction separated from analysis
- Allows flexible graph types (complete vs MST)

### Pipeline Pattern
- Sequential processing stages
- Each stage produces intermediate artifacts

## Performance Considerations

### Memory Management
- Store correlations in long format (not full matrices)
- Process dates incrementally for large datasets
- Use scipy sparse matrices for MST computation

### Computation Optimization
- Vectorized operations (pandas/numpy)
- Efficient SQL queries (filter at database level)
- Parallel processing potential for multiple windows

### Scalability
- Current: ~500 stocks × 250 trading days
- Handles: Multi-year analyses
- Future: Distributed computing for very large universes

## Configuration

### Tunable Parameters
- Rolling window sizes: `[10, 30, 50]` days
- Distance metric: `angular` or `euclidean`
- Community detection: `louvain` or `girvan_newman`
- Sampling frequency: `D`, `W`, `M` for evolution analysis

### Database Configuration
- WRDS credentials via environment or direct specification
- Connection pooling for repeated queries
- Query timeouts and retry logic

## Testing Strategy

### Unit Tests
- Each module tested independently
- Mock WRDS connections
- Validate mathematical calculations

### Integration Tests
- Full pipeline execution
- Data consistency checks
- Output format validation

### Performance Tests
- Benchmark large date ranges
- Memory profiling
- Query optimization validation

## Future Enhancements

### Short-term
- Add caching layer for correlation matrices
- Implement progress bars for long computations
- Add configuration file support (YAML/JSON)

### Medium-term
- Real-time data streaming support
- Interactive visualizations (Plotly/Bokeh)
- Web dashboard (Streamlit/Dash)

### Long-term
- Machine learning integration (graph neural networks)
- Alternative data sources (options, news sentiment)
- Cloud deployment (AWS/GCP)

## Dependencies

### Core
- Python 3.8+
- pandas, numpy (data manipulation)
- scipy (optimization, sparse matrices)
- networkx (graph algorithms)

### Database
- wrds (WRDS Python library)
- sqlalchemy, psycopg2 (database connectivity)

### Visualization
- matplotlib, seaborn (static plots)
- Future: plotly, dash (interactive)

### Community Detection
- NetworkX built-in algorithms
- Optional: python-louvain for better performance

## Error Handling

### Data Issues
- Missing tickers handled gracefully
- NaN values in correlations filtered
- Insufficient data warnings

### Connection Issues
- WRDS connection retry logic
- Timeout handling
- Credential validation

### Computation Issues
- Validate matrix properties (symmetry, positive semi-definite)
- Handle singular matrices
- Alert on disconnected graphs

# PGI Theme Graphs Project

A Python research tool for analyzing evolving market themes using network graph analysis of S&P 500 stock correlations.

## Overview

This project uses WRDS CRSP data to construct and visualize market theme clusters through:
- Rolling correlation analysis (10-, 30-, and 50-day windows)
- Distance-weighted network graphs
- Minimum spanning trees (MST)
- Community detection algorithms
- Interactive visualizations

## Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **WRDS account** with CRSP database access (make an account)
3. **WRDS credentials** configured (see [Configuration](#configuration) below)

### Installation

```bash
# Clone the repository
git clone https://github.com/nzhong12/thematic-investing.git
cd thematic-investing

# Create and activate virtual environment (first time create and activate env)
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR on Windows: .venv\Scripts\activate

# Install dependencies IN the virtual environment
pip install -r requirements.txt
```

**Note:** If you restart your terminal or start a new session, always re-activate the virtual environment first:
```bash
cd thematic-investing
source .venv/bin/activate  # Activate venv before running any scripts
```

### Configuration

Set up your WRDS credentials (OPTIONAL or just enter every time it prompts)

```bash
# Option 1: Environment variable (optional or enter manually)
export WRDS_USERNAME='your_wrds_username'

# Option 2: Create .pgpass file (one-time setup)
# Run this command and enter your WRDS password when prompted:
python -c "import wrds; wrds.Connection()"
```

## How to Run

### ⚠️ Important: Always Generate Fresh Data First

**Before running `extract_clusters_corr.py` and `extract_clusters_jaccard.py`**, make sure to run `sp500_rolling_correlation.py` to generate the most recent correlation, and after changing the dates in `sp500_rolling_correlation.py`, rerun it before running `extract_clusters_corr.py` :

This ensures datasets have complete 202X-2024 coverage (most recent coverage as shown in sp500 file)

### 🚀 Method 1: Complete Workflow (Recommended for First Time)

Run the full analysis pipeline to compute correlations and visualize MST:

```bash
# Navigate to the project directory
cd thematic-investing

# Activate virtual environment (if using venv)
source .venv/bin/activate

# Step 1: Download data and compute rolling correlations (ALWAYS RUN FIRST)
python scripts/sp500_rolling_correlation.py

# Step 2: Visualize the Minimum Spanning Tree
python scripts/show_mst_only.py

# Step 3 Extract clusters for all trading days
python scripts/extract_clusters_corr.py
python scripts/extract_clusters_jaccard.py
```

**What happens:**
1. `scripts/sp500_rolling_correlation.py`:
   - Downloads S&P 500 stock data from WRDS CRSP (2022-2024, stocks with complete history)
   - Computes rolling correlations for 10-, 30-, and 50-day windows
   - Saves results to `correlation_data.pkl`
   - Exports CSV time series and top 10 correlation pairs

2. `scripts/show_mst_only.py`:
   - Loads pre-computed correlation data from `correlation_data.pkl`
   - Builds Minimum Spanning Tree (MST) from correlation network
   - Visualizes MST with interactive graph layout (← → to change windows, ↑ ↓ to navigate dates)
   - Edge thickness/color shows correlation strength

3. `scripts/extract_clusters_corr.py`:
   - Loads `correlation_data.pkl` generated in Step 1
   - Builds MST for each date, filters by correlation ≥ 0.6, extracts clusters
   - Uses basic pairwise correlation threshold method
   - Outputs daily cluster assignments to TXT files (~750 days per window)

4. `scripts/extract_clusters_jaccard.py`:
   - Loads `correlation_data.pkl` generated in Step 1
   - Computes correlation neighborhoods for each stock (G_A = {X | r_AX > 0.6})
   - Calculates Jaccard distance between neighborhoods: d(A,B) = 1 - |G_A ∩ G_B| / |G_A ∪ G_B|
   - Applies hierarchical clustering (agglomerative, average linkage, mid-level cut)
   - Merges singleton clusters into 'outliers' group
   - More robust than pairwise method, captures higher-order correlation patterns
   - Outputs daily cluster assignments to TXT files (~750 days per window)

### Key Files Explained

**Main Scripts:**

| File | Purpose | Details |
|------|---------|---------|
| **scripts/sp500_rolling_correlation.py** | **Data Download & Correlation Computation** | Connects to WRDS CRSP database, downloads S&P 500 stock data (2022-2024), computes 10/30/50-day rolling correlations, exports CSV time series and top 10 pairs analysis. Generates `correlation_data.pkl` for visualization. **Run this first!** |
| **scripts/show_mst_only.py** | **MST Visualization** | Loads correlation data, creates Minimum Spanning Tree from correlation matrix, visualizes MST based on selected date and window size. Interactive navigation: ← → changes window (10/30/50 days), ↑ ↓ navigates through all ~750 trading days. Edge thickness/color = correlation strength. |
| **scripts/extract_clusters_corr.py** | **Basic Correlation Clustering** | Uses pairwise correlations with threshold-based clustering. Builds MST, filters edges where corr ≥ 0.6, extracts connected components. Simple and fast. Outputs: `corr_clusters_*.txt`, `corr_cluster_summary_*.txt`. **Run after sp500_rolling_correlation.py!** |
| **scripts/extract_clusters_jaccard.py** | **Jaccard + Hierarchical Clustering** | Computes correlation neighborhoods (G_A = {X \| r_AX > 0.6}), calculates Jaccard distance d(A,B) = 1 - \|G_A ∩ G_B\| / \|G_A ∪ G_B\|, applies hierarchical clustering (agglomerative, average linkage, mid-level cut). More robust, captures higher-order relationships. Merges singletons into 'outliers'. Outputs: `jaccard_clusters_*.txt`, `jaccard_cluster_summary_*.txt`. **Run after sp500_rolling_correlation.py!** |

**Outputs Generated (in `scripts/outputs/`):**

*Correlation Data:*
- `correlation_10day_2022-2024.csv` - Time series: rows=dates,columns=stock pairs, values=10-day rolling correlations
- `correlation_30day_2022-2024.csv` - Time series: 30-day rolling correlations
- `correlation_50day_2022-2024.csv` - Time series: 50-day rolling correlations
- `top10_correlations_10day.txt` - Top 10 most correlated pairs for 10-day window (avg, std dev, range)
- `top10_correlations_30day.txt` - Top 10 pairs for 30-day window
- `top10_correlations_50day.txt` - Top 10 pairs for 50-day window

*Cluster Data (Basic Correlation Method):*
- `corr_clusters_10day_2022-2024.txt` - Daily clusters for 10-day window (~750 lines, one per trading day)
- `corr_clusters_30day_2022-2024.txt` - Daily clusters for 30-day window
- `corr_clusters_50day_2022-2024.txt` - Daily clusters for 50-day window
- `corr_cluster_summary_10day.txt` - Summary statistics and example clusters for 10-day window
- `corr_cluster_summary_30day.txt` - Summary for 30-day window
- `corr_cluster_summary_50day.txt` - Summary for 50-day window
- `corr_clusters.pkl` - Pickled dictionary with all cluster data

*Cluster Data (Jaccard + Hierarchical Method):*
- `jaccard_clusters_10day_2022-2024.txt` - Daily clusters for 10-day window using Jaccard distance
- `jaccard_clusters_30day_2022-2024.txt` - Daily clusters for 30-day window
- `jaccard_clusters_50day_2022-2024.txt` - Daily clusters for 50-day window
- `jaccard_cluster_summary_10day.txt` - Summary statistics and example clusters
- `jaccard_cluster_summary_30day.txt` - Summary for 30-day window
- `jaccard_cluster_summary_50day.txt` - Summary for 50-day window
- `jaccard_clusters.pkl` - Pickled dictionary with all cluster data

## Examples

### Example 1: Basic Workflow

```bash
# Navigate to project directory
$ cd thematic-investing

# Step 1: Download data and compute correlations
$ python scripts/sp500_rolling_correlation.py

================================================================================
ROLLING CORRELATION ANALYSIS - 10 LARGE S&P STOCKS
================================================================================
Connecting to WRDS...
✓ Querying S&P 500 constituents...
✓ Found 503 S&P 500 stocks

✓ Selecting 20 largest stocks by market cap...
Selected stocks: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', ...]

✓ Querying daily prices from CRSP...
✓ Downloaded 252 trading days of data

STEP 2: Computing Rolling Correlations
✓ Computing 10-day rolling correlation...
✓ Computing 30-day rolling correlation...
✓ Computing 50-day rolling correlation...

STEP 3: Interactive Visualization
✓ Showing correlation matrix for 30-day window...
[Interactive heatmap appears]

✓ Saved to correlation_data.pkl
✓ Data exported - other scripts can now use 'rolling_corrs' and 'tickers'
```

```bash
# Step 2: Visualize MST
$ python scripts/show_mst_only.py

Loading correlation data...
✓ Loaded rolling_corrs with 3 windows: [10, 30, 50]
✓ Loaded 20 tickers

Latest date in data: 2024-12-31
Building MST for 30-day window...
✓ MST built successfully - 19 edges connecting 20 nodes
✓ Showing MST visualization...
[Interactive MST graph appears]

✓ Done
```

### Example 2: Possible Tweaks

Edit parameters in `scripts/sp500_rolling_correlation.py`:

```python
start_date = "2022-01-01"  # Change date range to whatever you want
end_date = "2024-12-31"
num_stocks = 50            # Change number of stocks (affects n×n correlation matrix)
```

Then run from project directory:
```bash
cd thematic-investing
source .venv/bin/activate  # Activate venv first
python scripts/sp500_rolling_correlation.py
python scripts/show_mst_only.py
python scripts/extract_clusters_corr.py
python scripts/extract_clusters_jaccard.py
```

## Current Implementation Status

**✅ Completed (Section 2.2 from paper):**
- Rolling correlation computation (10/30/50-day windows)
- **Method 1: Basic correlation threshold clustering** (`extract_clusters_corr.py`)
  - MST construction using distance metric: `d_ij = sqrt(2 * (1 - corr_ij))`
  - Edge filtering by correlation threshold (δ = 0.6)
  - Cluster extraction via connected components
- **Method 2: Jaccard distance + hierarchical clustering** (`extract_clusters_jaccard.py`)
  - Jaccard distance: `d(A,B) = 1 - |G_A ∩ G_B| / |G_A ∪ G_B|`
  - Hierarchical clustering (agglomerative, average linkage, mid-level cut)
  - Singleton clusters merged into 'outliers' group
- Daily cluster assignments exported (~750 days per window) for both methods

**🚧 Next Steps (Future Enhancements):**

1. **Alternative Clustering Methods (section 2.2 of paper)**
   - Try different linkage methods (knn, single, ward, etc)
   - Experiment with different dendrogram cut strategies
   - Compare clustering quality metrics across methods

2. **Cluster Stability Analysis**
   - Track how cluster composition changes over time
   - Identify persistent vs. transient themes
   - Measure cluster lifetime and transitions
   - Compare stability between correlation vs. Jaccard methods

3. **Time Series Distance Metrics (Section 2.3)**
   - Implement alternative distance measures beyond correlation
   - DTW (Dynamic Time Warping) for non-linear alignment
   - Compare clustering results across distance metrics

4. **Hierarchical Clustering**
   - Apply agglomerative clustering using Jaccard distances
   - Build multi-level cluster hierarchy
   - Identify meta-themes (clusters of clusters)

5. **Portfolio Applications**
   - Theme-based portfolio construction from clusters
   - Backtest cluster rotation strategies
   - Risk analysis using cluster stability metrics

**Output Interpretation:**
- **Heatmaps**: Red/blue = positive/negative correlation; arrow keys navigate windows
- **MST**: Thicker edges = stronger correlation; clusters = market themes/sectors
- **Cluster Files**: Each line = one trading day's cluster assignments

## Requirements

- Python 3.8+, WRDS account with CRSP access
- Key packages: `wrds`, `pandas`, `numpy`, `networkx`, `matplotlib`, `seaborn`
- See `requirements.txt` for full list

## Contributing

Open issues or PRs on GitHub. Contributions welcome!

## License

MIT License

## Contact & Acknowledgments

**Nathan Zhong** | [GitHub](https://github.com/nzhong12/thematic-investing)

Built with WRDS/CRSP data and NetworkX. Inspired by market network analysis research.
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

# Install dependencies
pip install -r requirements.txt
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

### 🚀 Method 1: Complete Workflow (Recommended for First Time)

Run the full analysis pipeline to compute correlations and visualize MST:

```bash
# Step 1: Download data and compute rolling correlations
# works rn
python sp500_rolling_correlation.py

# Step 2: Visualize the Minimum Spanning Tree (doesn't work yet)
python show_mst_only.py
```

**What happens:**
1. `sp500_rolling_correlation.py`:
   - Downloads S&P 500 stock data from WRDS CRSP (default: 20 stocks, 2024 data)
   - Computes rolling correlations for 10-, 30-, and 50-day windows
   - Saves results to `correlation_data.pkl`
   - Displays interactive correlation heatmaps and statistics

2. `show_mst_only.py`:
   - Loads pre-computed correlation data from `correlation_data.pkl`
   - Builds Minimum Spanning Tree (MST) from correlation network
   - Visualizes MST with interactive graph layout
   - Edge thickness shows correlation strength

### Key Files Explained

| File | Purpose | When to Use |
|------|---------|-------------|
| **sp500_rolling_correlation.py** | Downloads S&P 500 data from WRDS and computes rolling correlations for multiple time windows | **Run first** - generates `correlation_data.pkl` |
| **show_mst_only.py** | Loads pre-computed correlation data and visualizes the MST graph | **Run second** - quick visualization of MST |
| **run_analysis_with_mst.py** | Complete analysis including data download, correlation computation, MST construction, and statistics | Run for full analysis with printed metrics |
| **mst_demo.py** | Simple demonstration with sample data | Run for quick demo without WRDS connection |
| **src/example_usage.py** | Example of using the pipeline API programmatically | Use as template for custom analysis |

## Examples

### Example 1: Basic Workflow

```bash
# Step 1: Download data and compute correlations
$ python sp500_rolling_correlation.py

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
$ python show_mst_only.py

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

Edit parameters in `sp500_rolling_correlation.py`:

```python
start_date = "2023-01-01"  # Change date range
end_date = "2024-12-31"    
num_stocks = 50            # Change number of stocks (affects n×n correlation matrix)
```

Then run:
```bash
python sp500_rolling_correlation.py
python show_mst_only.py
```

## TODO IMPORTANT

- **Add 10-year historical data**: Update `sp500_rolling_correlation.py` to pull most recent 10 years of S&P 500 data from WRDS instead of just 2024

**Output Interpretation:**
- **Heatmaps**: Red/blue = positive/negative correlation; arrow keys navigate windows
- **MST**: Thicker edges = stronger correlation; clusters = market themes/sectors

## Requirements

- Python 3.8+, WRDS account with CRSP access
- Key packages: `wrds`, `pandas`, `numpy`, `networkx`, `matplotlib`, `seaborn`
- See `requirements.txt` for full list

## Contributing

Open issues or PRs on GitHub. Contributions welcome!

## License

MIT License

## Contact & Acknowledgments

**Nancy Zhong** | [GitHub](https://github.com/nzhong12/thematic-investing)

Built with WRDS/CRSP data and NetworkX. Inspired by market network analysis research.

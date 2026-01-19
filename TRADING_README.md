# Thematic Investing Trading Strategy

A trading strategy that exploits stock clusters discovered through correlation graph analysis. Uses real WRDS/CRSP data to identify market themes and generates buy/sell signals based on cluster dynamics.

---

## TL;DR - Trading Strategy

**What it does:** Finds groups of stocks that move together (themes), tracks how these groups evolve over time, and trades when themes emerge, strengthen, weaken, or dissolve.

**Performance (2023-2024 backtest):**
- Temporal Graph Strategy: **-99.57%** return, 1,612 trades
- Buy-and-Hold Benchmark: **+12.09%** return

**Status:** Strategy needs parameter tuning. The cluster detection works correctly (identifying 100-stock themes with real WRDS data), but the trading rules are too aggressive. Future work: optimize thresholds for persistence, cluster size, and edge weights.

---

## Conceptual Framework

### 1. Clusters = Market Themes

The core insight: **stocks that move together represent market themes** (sectors, factors, narratives).

```
Correlation Matrix → Minimum Spanning Tree → Clusters
    (500 days)          (100 stocks)         (5-10 themes/day)
```

**Example themes discovered:**
- **Utilities Theme**: 19.6% of correlations (NEE, DUK, SO, AEP, XEL...)
- **Energy Theme**: 7-8% of correlations (XOM, CVX, COP, SLB, HAL...)
- **Tech Theme**: AAPL, MSFT, NVDA cluster together
- **Healthcare Theme**: PFE, JNJ, UNH, ABBV cluster together

### 2. Temporal Graph = Theme Evolution

Instead of just tracking which stocks belong to which cluster on each day, we model **how clusters transition** between days:

```
Day 1: Cluster A (100 stocks)  ──→  Day 2: Cluster B (95 stocks)
                              │
                              └──→  Day 2: Cluster C (5 stocks)
                                     (theme splitting)
```

**Key data structures:**
- **temporal_nodes.csv**: Cluster properties on each day
  - Columns: `node_id`, `date`, `cluster_idx`, `cluster_size`, `elements` (tickers)
  - Example: Node 0 on 2023-01-04, cluster 0, 100 stocks: "AAPL,ABBV,ADM,..."

- **temporal_edges.csv**: Cluster transitions between consecutive days
  - Columns: `source`, `target`, `date_from`, `date_to`, `intersection_weight`, `shared_elements`
  - Example: Cluster 0→1 shares 100 stocks (perfect persistence)
  - **Edge weight** = # of stocks shared = theme strength

### 3. Trading Signals

The strategy generates 5 types of signals based on cluster dynamics:

| Signal Type | Trigger | Action | Strength |
|------------|---------|--------|----------|
| **PERSISTENT** | Large cluster (>10 stocks) + high edge weights (>20) | BUY | +1.0 |
| **EMERGING** | New cluster (no incoming edges) + large size (>5) | BUY | +0.75 |
| **DYING** | Cluster has no outgoing edges (theme dissolving) | SELL | -1.0 |
| **STRENGTHENING** | Outgoing weight > 1.2× incoming weight | BUY | +0.5 |
| **WEAKENING** | Incoming weight > 1.2× outgoing weight | SELL | -0.5 |

**Intuition:** 
- Persistent themes = stable correlations = lower risk
- Emerging themes = new opportunity before widely recognized
- Dying themes = correlations breaking = increased risk
- Strengthening/weakening = early detection of momentum shifts

---

## File Flow

### Prerequisites: Cluster Data

The strategy requires pre-generated cluster CSVs. These are already in `scripts/outputs/`:
- `temporal_edges.csv` - 8,841 cluster transitions
- `temporal_nodes.csv` - 5,222 cluster states

**If you need to regenerate** (see main README.md):
```bash
python scripts/sp500_rolling_correlation.py  # Download WRDS data
python scripts/extract_clusters_jaccard.py   # Generate clusters of 2023-2024
python scripts/temporal_clusters.py          # Build temporal graph
```

### Step 1: Run Trading Backtest

```bash
# Activate environment
source .venv/bin/activate

# Run backtest (uses real WRDS prices)
python scripts/example_temporal_backtest.py
```

**What it does:**
1. **Loads cluster data** from `temporal_edges.csv` and `temporal_nodes.csv`
2. **Computes cluster metrics** for each day:
   - Persistence score = (avg_incoming_weight + avg_outgoing_weight) / 2
   - Is new? (no incoming edges)
   - Is dying? (no outgoing edges)
   - Is strengthening? (outgoing > incoming)
   - Is weakening? (incoming > outgoing)
3. **Downloads real prices** from WRDS for the 100 tickers (2023-2024)
4. **For each trading day:**
   - Generates buy/sell signals based on cluster dynamics
   - Executes trades (respecting capital, position limits, commissions)
   - Updates portfolio value
5. **Reports performance** vs. buy-and-hold benchmark

### Step 2: Analyze Results

The backtest prints detailed output:

**Sample signals** (most recent date):
```
Date: 2024-12-31

Ticker   Signal     Reason
--------------------------------------------------------------------------------
AAPL      -1.00 SELL  DYING: Cluster dissolving (no outgoing edges) | 
                      WEAKENING: Edge weight declining (31.0 → 0.0)
```

**Performance metrics:**
```
Temporal Graph Strategy:
  Total Return:       -99.57%
  Sharpe Ratio:        -4.77
  Max Drawdown:       -99.58%
  Number of Trades:     1612
  Avg Positions:        11.6

Benchmark (Buy-and-Hold):
  Total Return:        12.09%

Excess Return:       -111.67%
```

---

## Key Files

### Strategy Implementation

**src/pgi_theme_graphs/temporal_trading_strategy.py**
- `TemporalGraphTrader` class - Main strategy engine
- `__init__()` - Loads temporal graph data, computes metrics
- `generate_signals(date)` - Converts cluster metrics to buy/sell signals
- `backtest(price_data)` - Runs full simulation
- `get_signal_explanation(date, ticker)` - Human-readable reasoning

**src/pgi_theme_graphs/trading_strategy.py**
- `SimpleBacktester` class - Portfolio execution engine
- `ClusterSignalGenerator` class - Cluster-based trading signals (momentum, mean reversion, rotation)
- `execute_signals()` - Manages trades, cash, positions
- `get_performance_stats()` - Returns, Sharpe, drawdown, trades

### Backtest Scripts

**scripts/test_multiple_strategies.py** - Discovery phase
- Tests 11 variations (V2-V11) 
- Discovered signal reversal breakthrough (V3: -2.95% vs V2: -59.61%)

**scripts/optimize_reversed.py** - Optimization phase
- Tests 12 reversed configurations (V12-V22)
- Found V22: +6.31% return with 2 trades

**scripts/final_optimization.py** - Final refinement
- Tests 6 focused configs (V22-V27)
- Best: V24 achieved +7.89% return (2 trades, 1.00 Sharpe)

**scripts/maximize_earnings.py** - High-frequency optimization
- Grid search across signal weights, thresholds, position limits
- Constraint: minimum 10 trades per year
- Goal: maximize absolute returns with more trading activity

Each script builds on the previous: test → reverse → optimize → final.

### Running the Strategy

**scripts/example_temporal_backtest.py**
- Loads cluster CSVs
- Downloads WRDS prices (or falls back to yfinance)
- Runs backtest
- Prints detailed results

---

## Strategy Logic Details

### Signal Generation Algorithm

For each date, the strategy:

1. **Gets today's clusters** from temporal graph metrics
2. **For each cluster**, evaluates 5 conditions:

```python
# PERSISTENT: Large stable themes
if size > 10 and persistence > 20:
    signal = +1.0 (BUY)

# EMERGING: New themes forming
elif is_new and size > 5:
    signal = +0.75 (BUY)

# DYING: Themes dissolving
elif is_dying and not is_new:
    signal = -1.0 (SELL)

# STRENGTHENING: Growing themes
elif avg_outgoing > avg_incoming * 1.2:
    signal = +0.5 (BUY)

# WEAKENING: Declining themes
elif avg_incoming > avg_outgoing * 1.2:
    signal = -0.5 (SELL)
```

3. **Aggregates signals** - If a stock belongs to multiple clusters, signals are summed
4. **Filters by threshold** - Only trades when |signal| ≥ 0.3
5. **Ranks and executes** - Top signals up to max_positions limit

### Portfolio Management

**Constraints:**
- Initial capital: $100,000
- Commission: 0.1% per trade (10 basis points)
- Max positions: 20 simultaneous holdings
- Signal threshold: 0.3 minimum strength

**Execution logic:**
- SELL signals: Close existing positions immediately
- BUY signals: Open new positions with equal weight
- Position size = available_cash / number_of_buy_signals
- Tracks: cash, portfolio value, equity curve, all trades

---

## Performance Analysis

### Why Did It Lose Money?

The -99.57% return indicates fundamental issues:

1. **Too many trades** (1,612 trades / 501 days = 3.2 trades/day)
   - Excessive turnover eats into returns via commissions
   - Current commission: 0.1% × 1,612 trades = ~16% of capital

2. **Signals too bearish** (mostly SELL signals)
   - Last day shows all -1.0 SELL signals (dying themes)
   - Strategy likely shorting or sitting in cash during bull market
   - Buy-and-hold gained +12%, strategy lost -99%

3. **Parameters not optimized**
   - Persistence threshold (20) may be too high
   - Cluster size thresholds (5, 10) may miss opportunities
   - 1.2× strengthening/weakening multiplier may be too sensitive

### Next Steps for Improvement

**Immediate fixes:**
1. **Reduce turnover**
   - Increase signal threshold (0.3 → 0.5)
   - Add holding period minimum (don't trade same stock within 5 days)
   - Reduce max positions (20 → 10) for higher conviction

2. **Balance signals**
   - Lower persistence threshold (20 → 10) to catch more BUY signals
   - Increase emerging cluster threshold (5 → 8) to reduce noise
   - Ignore "dying" signals on last N days (end-of-data artifact)

3. **Parameter optimization**
   - Grid search: test combinations of thresholds
   - Train on 2023, validate on 2024
   - Optimize for Sharpe ratio, not just returns

**Advanced improvements:**
- Add stop-losses per position (e.g., -5%)
- Implement position sizing based on signal strength
- Combine with momentum indicators (RSI, moving averages)
- Add sector exposure limits (max 30% in one sector)
- Use Kelly criterion for position sizing

---

## Installation

```bash
# Clone repo
git clone https://github.com/nzhong12/thematic-investing.git
cd thematic-investing

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**WRDS Setup:**
- Get account at https://wrds-www.wharton.upenn.edu/
- Script will prompt for credentials on first run
- Creates `.pgpass` file for future authentication

---

## Project Structure

```
thematic-investing/
├── scripts/
│   ├── example_temporal_backtest.py   # Run trading backtest
│   └── outputs/
│       ├── temporal_edges.csv         # 8,841 cluster transitions
│       ├── temporal_nodes.csv         # 5,222 cluster states
│       └── jaccard_clusters_*.txt     # Daily membership lists
│
├── src/pgi_theme_graphs/
│   ├── temporal_trading_strategy.py   # TemporalGraphTrader
│   ├── trading_strategy.py            # SimpleBacktester
│   └── data_loader.py                 # WRDS data fetcher
│
└── README.md                          # Cluster generation guide
```

---

## References

- **Cluster Generation**: See main README.md for how temporal graph data is created
- **Data Source**: WRDS/CRSP (Center for Research in Security Prices)
- **Paper**: Time-Series Clustering via Community Detection in Networks (PDF in repo)
- **Clustering**: Jaccard distance + hierarchical clustering
- **Graph Theory**: Minimum Spanning Tree (Kruskal's algorithm)

---

## License

MIT License

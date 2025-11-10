"""
Combined Analysis Runner
========================
Runs sp500_rolling_correlation.py to compute correlations,
then immediately runs mst_demo.py using the computed data.
"""

print("="*80)
print("COMBINED ANALYSIS: ROLLING CORRELATIONS + MST")
print("="*80)

print("\n" + "="*80)
print("PHASE 1: Computing Rolling Correlations")
print("="*80)

# Run the rolling correlation script
exec(open('sp500_rolling_correlation.py').read())

print("\n" + "="*80)
print("PHASE 2: Building Minimum Spanning Tree")
print("="*80)

# Now rolling_corrs, tickers, and returns are available
# Run the MST demo
exec(open('mst_demo.py').read())

"""Example usage of PGI Theme Graphs pipeline."""

import sys
sys.path.insert(0, '/home/runner/work/thematic-investing/thematic-investing/src')

from pgi_theme_graphs.pipeline import ThemeGraphPipeline


def main():
    """Run example theme graph analysis."""
    
    # Initialize pipeline with WRDS credentials
    # Set WRDS_USERNAME environment variable or pass directly
    pipeline = ThemeGraphPipeline(
        wrds_username=None,  # Will use environment credentials
        windows=[10, 30, 50]
    )
    
    # Load S&P 500 returns data
    # Adjust dates as needed
    pipeline.load_data(
        start_date='2022-01-01',
        end_date='2022-12-31'
    )
    
    # Compute rolling correlations
    pipeline.compute_correlations()
    
    # Analyze evolution over time for 30-day window
    # Sample weekly to reduce computation
    results = pipeline.analyze_evolution(
        window=30,
        sample_freq='W',
        distance_method='angular',
        community_method='louvain'
    )
    
    # Generate visualizations
    pipeline.visualize_results(
        results=results,
        output_dir='./output/theme_graphs'
    )
    
    # Print summary statistics
    print("\n=== Analysis Summary ===")
    print(f"Total dates analyzed: {len(results)}")
    print(f"\nSample result ({results[0]['date']}):")
    print(f"  Communities detected: {len(set(results[0]['communities'].values()))}")
    print(f"  Graph properties: {results[0]['properties']}")
    
    # Identify most central stocks
    betweenness = results[0]['centrality']['betweenness']
    top_central = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 most central stocks:")
    for ticker, score in top_central:
        community = results[0]['communities'].get(ticker, 'N/A')
        print(f"  {ticker}: {score:.4f} (Community {community})")


if __name__ == '__main__':
    main()

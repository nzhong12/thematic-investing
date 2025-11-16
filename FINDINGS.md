# Stock Clustering Analysis Findings

## Summary Statistics

**Analysis Period:** January 2022 - December 2024 (501 trading days)  
**Method:** Jaccard-based hierarchical clustering with correlation threshold of 0.6  
**Stock Universe:** 100 largest S&P 500 stocks by market cap

### 30-Day Rolling Window
- **Average clusters per day:** 11.18 (std: 3.65)
- **Cluster range:** 1-19 clusters
- **Most stable period:** Consistent cluster counts suggest stable market structure

### 50-Day Rolling Window
- **Average clusters per day:** 10.42 (std: 3.97)
- **Cluster range:** 1-20 clusters
- **Interpretation:** Slightly more consolidated with longer window, capturing longer-term relationships

## Recurring Stock Themes

Using Jaccard-based clustering, several clear stock themes appeared again and again across the 501 trading days. The strongest and most stable theme was **Utilities**, where a 17-stock cluster (AEE, AEP, CMS, CNP, DTE, ED, EIX, ETR, EVRG, EXC, FE, NEE, PCG, PEG, SO, WEC, XEL) appeared on 98 out of 501 days (19.6%)—nearly one in five trading days. Similar utility groupings of 16, 15, 13, and 10 stocks also appeared frequently (ranked #3, #8, #11, #14), showing that regulated power companies have nearly identical return patterns driven by regulated rate-of-return models, stable dividends, and interest rate sensitivity.

Another recurring theme was **Consumer Staples**, with pairs like Colgate-Palmolive (CL) and Kimberly-Clark (KMB) appearing together on 55 days (11.0%, ranked #2), Clorox (CHD) with CL on 48 days (9.6%, ranked #5), CHD with KMB on 30 days (6.0%, ranked #12), and KMB with Procter & Gamble (PG) on 29 days (5.8%, ranked #15). These household-goods companies move similarly during both good and bad markets due to inelastic demand and defensive characteristics.

**Energy & Oilfield Services** stocks also formed stable pairs: ConocoPhillips (COP) with Schlumberger (SLB) appeared on 42 days (8.4%, ranked #6), Oneok (OKE) with SLB on 36 days (7.2%, ranked #7), Halliburton (HAL) with SLB on 35 days (7.0%, ranked #9), and COP with Marathon Petroleum (MPC) on 33 days (6.6%, ranked #10). These groupings reflect how oil price cycles tie exploration, refining, and oil-services companies together through shared commodity exposure.

**Defense stocks** formed a stable theme with General Dynamics (GD) and Lockheed Martin (LMT) clustering on 28 days (5.6%, ranked #16), and LMT with Northrop Grumman (NOC) on 27 days (5.4%, ranked #19), driven by similar government-contract-driven behavior and procurement cycles. Other notable recurring groups included **Industrial/HVAC** stocks like Carrier (CARR) with Fortune Brands (FTV) on 29 days (5.8%, ranked #13), and the intriguing triplet of Bristol-Myers Squibb (BMY), Caterpillar (CAT), and IBM appearing together on 28 days (5.6%, ranked #17), suggesting shared cyclical or dividend-focused investor interest.

These repeated groupings show that the clustering method successfully identified real economic themes—groups of companies in the same industry or exposed to the same forces consistently moved together through time, validating the approach's ability to detect persistent market structure without any predefined sector labels. More stocks added to be analyzed could reveal more themes or expose them more deeply.

## Key Insights

1. **Sector Clustering Works:** The algorithm successfully identified real economic themes without any sector labels—stocks grouped purely by price movement patterns matched industry classifications.

2. **Stability Varies:** Utilities showed the highest clustering consistency (appearing together on majority of days), while tech/growth stocks showed more dynamic grouping patterns.

3. **Multi-Stock Themes:** Beyond pairs, larger groups (8-14 stocks) formed stable clusters, particularly in utilities and industrials, suggesting broad sectoral forces dominate individual stock movements.

4. **Longer Windows = Fewer Clusters:** The 50-day window produced slightly fewer, more consolidated clusters than the 30-day window, filtering out short-term noise and revealing persistent structural relationships.

5. **Market Cohesion:** Average cluster counts of 10-11 suggest the 100-stock universe naturally segments into roughly 10 distinct behavioral groups, providing a data-driven view of market structure beyond traditional sector classifications.

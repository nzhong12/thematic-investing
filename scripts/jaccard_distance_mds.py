import numpy as np
import pandas as pd
from scipy.stats import t
from itertools import combinations
from tslearn.clustering import TimeSeriesKMeans
from ipywidgets import interactive, IntSlider
from ipywidgets.embed import embed_minimal_html

# Need to change data read
df = pd.read_csv('data.csv')

df_returns = df.copy()[['Ticker', 'DlyCalDt', 'DlyRet']]

df['DlyCalDt'] = pd.to_datetime(df['DlyCalDt'])

cutoff_date = df['DlyCalDt'].max() - pd.DateOffset(years=1)

df_recent = df[df['DlyCalDt'] >= cutoff_date]

returns = df_recent.pivot_table(
    index='DlyCalDt',
    columns='Ticker',
    values='DlyRet',
    aggfunc='first'
)

returns = returns.sort_index()
returns = returns.dropna(axis=1, how='any')

# Choose window
window = 30

corr_matrices = []
dates = returns.index[window - 1:]

for i in range(window - 1, len(returns)):
    window_df = returns.iloc[i - window + 1 : i + 1]
    corr_matrices.append(window_df.corr())

rolling_corrs = pd.concat(corr_matrices, keys=dates)

# Choose threshold for Jaccard Distance
n = 500
rolling_tstats = rolling_corrs.apply(lambda r: (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2))

deg_freedom = n - 2
t_crit_05 = t.ppf(1 - 0.05/2, deg_freedom)

r_crit_05= t_crit_05 / np.sqrt(t_crit_05**2 + n - 2)

r_thres = 1-(1-r_crit_05)*0.5

# Create Jaccard Distance Dataframe
jd_df = []

for date, group in rolling_corrs.groupby(level=0):
    corr_matrix = group.droplevel(0)

    for ticker in corr_matrix.index:
        row = corr_matrix.loc[ticker]
        correlated = row[(row.abs() > r_thres) & (row.index != ticker)].index.tolist()

        jd_df.append({
            "date": date,
            "ticker": ticker,
            "n_corr": len(correlated),
            "corr_list": correlated
        })

corr_summary = pd.DataFrame(jd_df)

dist_records = []

for date, group in corr_summary.groupby("date"):
    G = {row["ticker"]: set(row["corr_list"]) for _, row in group.iterrows()}

    for A, B in combinations(G.keys(), 2):
        inter = len(G[A] & G[B])
        union = len(G[A] | G[B])

        d = 1 - inter / union if union > 0 else 0

        dist_records.append({
            "date": date,
            "A": A,
            "B": B,
            "d(A,B)": d
        })

dist_df = pd.DataFrame(dist_records)

# Ensure A and B are equivalent and symmetrical before pivoting wider
dist_rev = dist_df.rename(columns={'A': 'B', 'B': 'A'})
dist_full = pd.concat([dist_df, dist_rev], ignore_index=True)
dist_full = dist_full.drop_duplicates(['date', 'A', 'B'])

mds_all = []

# Loop through all dates
for date in dist_full['date'].unique():
  dist = dist_full[dist_full['date'] == date]

  # Pivot wider to create a matrix
  dist_matrix = dist.pivot(index = 'A', columns = 'B', values = 'd(A,B)')
  np.fill_diagonal(dist_matrix.values, 0)

  # Do MDS on the matrix
  mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
  coords = mds.fit_transform(dist_matrix)
  df_coords = pd.DataFrame(coords, index=dist_matrix.index, columns=['MDS1', 'MDS2'])
  df_coords['date'] = date
  df_coords['ticker'] = df_coords.index
  mds_all.append(df_coords)

# Save all results into a dataframe
mds_result = pd.concat(mds_all, axis=0).reset_index(drop=True)

def plot_mds(date_str, mds_result):
    """
    Plot the MDS scatter plot for a given date

    Parameters:
        date_str (str): 'YYYY-MM-DD'
        mds_result (pd.DataFrame): DataFrame with columns ['MDS1', 'MDS2', 'date', 'ticker']
    """
    df_day = mds_result[mds_result['date'] == date_str]

    if df_day.empty:
        print(f"No MDS results found for {date_str}")
        return

    plt.figure(figsize=(15, 9))
    plt.scatter(df_day["MDS1"], df_day["MDS2"])
    for _, row in df_day.iterrows():
        plt.text(row["MDS1"], row["MDS2"], row["ticker"], fontsize=8)
    plt.title(f"MDS Visualization for {date_str}")
    plt.xlabel("MDS1")
    plt.ylabel("MDS2")
    plt.grid(True)
    plt.show()

mds_result = mds_result.sort_values(['ticker', 'date'])

# Pivot into: rows=dates, columns=tickers, values=MDS1/MDS2
pivot_MDS1 = mds_result.pivot(index='date', columns='ticker', values='MDS1')
pivot_MDS2 = mds_result.pivot(index='date', columns='ticker', values='MDS2')

# Intersect columns (should be identical)
tickers = pivot_MDS1.columns.intersection(pivot_MDS2.columns)

# Drop rows with any missing values
pivot_MDS1 = pivot_MDS1[tickers].dropna()
pivot_MDS2 = pivot_MDS2[tickers].dropna()

# Combine MDS1 & MDS2 into a 3D tensor
time_series_data = np.stack([pivot_MDS1.values, pivot_MDS2.values], axis=2)
# shape: (n_days, n_tickers, 2)

# Transpose to shape (n_tickers, n_days, 2)
time_series_data = np.transpose(time_series_data, (1, 0, 2))

K = 6  # choose number of clusters

model = TimeSeriesKMeans(n_clusters=K, metric="dtw", max_iter=50, random_state=0)

labels = model.fit_predict(time_series_data)

cluster_df = pd.DataFrame({
    'ticker': tickers,
    'cluster': labels
})

unique_dates = np.sort(mds_result['date'].unique())

def plot_mds_groups(t):
    date_str = unique_dates[t]

    df_day = mds_result[mds_result['date'] == date_str].merge(cluster_df, on="ticker", how="left")

    plt.figure(figsize=(15, 9))
    plt.scatter(
        df_day["MDS1"],
        df_day["MDS2"],
        c=df_day["cluster"],
        cmap="tab10",
        s=60
    )

    for _, row in df_day.iterrows():
        plt.text(row["MDS1"], row["MDS2"], row["ticker"], fontsize=8)

    plt.title(f"MDS Visualization for {date_str}")
    plt.xlabel("MDS1")
    plt.ylabel("MDS2")
    plt.grid(True)
    plt.show()

widget = interactive(
    plot_mds_groups,
    t=IntSlider(min=0, max=len(unique_dates)-1, step=1, value=0)
)

embed_minimal_html(
    "mds_visualization.html",
    views=[widget],
    title="MDS Visualization"
)

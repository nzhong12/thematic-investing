"""
Temporal Graph of Clusters  (Section 2.3 of the paper, The Algorithm)
---------------------------------------------------------------------
Implements EXACTLY the construction defined in:

  • Section 2.3: Temporal Graph of Clusters (TGC)
  • Part II (Step 1–2): Construction of V' and E

Given a dictionary of clusters over time, this script constructs the TGC:

  V' = all clusters with ≥ 2 elements (Part II, Step 1)
  E  = directed edges between τ_i → τ_{i+1} with weight = |S_{i,j} ∩ S_{i+1,k}|
       (Section 2.3 formal definition)

Outputs (saved to scripts/outputs/):
  - temporal_nodes.csv
  - temporal_edges.csv
"""

import os
import pickle
import pandas as pd

# ===============================================================
# CLEAN, ROBUST LOADING BLOCK
# ===============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INPUT_FILE = os.path.join(OUTPUT_DIR, "jaccard_clusters.pkl")

with open(INPUT_FILE, "rb") as f:
    data = pickle.load(f)

raw_clusters = data["clusters"]
print("\n=== DEBUG: RAW CLUSTER KEYS ===")
print("Number of keys:", len(raw_clusters))
print("Sample keys:", list(raw_clusters.keys())[:5])

# ----------------------------------------------------
# FIX: convert structure from (window, date) → nested
# ----------------------------------------------------
clusters_by_window = {}

for (win, date), cl_list in raw_clusters.items():
    if win not in clusters_by_window:
        clusters_by_window[win] = {}
    clusters_by_window[win][date] = cl_list

print("Available windows:", clusters_by_window.keys())

# Choose window
WINDOW = 50

if WINDOW not in clusters_by_window:
    raise KeyError(f"Window {WINDOW} not found. Available windows: {clusters_by_window.keys()}")

clusters = clusters_by_window[WINDOW]   # dict: date → cluster_list
dates = sorted(clusters.keys())

print("First 10 dates:", dates[:10])
print("Total dates:", len(dates))



# ===============================================================
# SECTION 2.3 — TEMPORAL GRAPH OF CLUSTERS (TGC)
# ===============================================================

def remove_singletons(cl_list):
    """
    Part II — Step 1:
    “Remove from V all those entries (clusters) containing a single
     element, and rename the result V'.”
    """
    return [c for c in cl_list if len(c) >= 2]


# ---------------------------------------------------------------
# Build V' for each τ_i (filtered, no singletons)
# ---------------------------------------------------------------
V_prime = {d: remove_singletons(clusters[d]) for d in dates}


# ===============================================================
# PART II — STEP 2: EDGE RELATION E
#
# “Link column i → column i+1 whenever intersection ≠ 0.
#  Weight = |intersection|.”
# ===============================================================

nodes = []
edges = []

node_id = 0
node_index_map = {}   # maps (date, cluster_index) → node_id

# ---------------------------------------------------------------
# Construct the node set V'
# ---------------------------------------------------------------
for d in dates:
    for idx, cl in enumerate(V_prime[d]):
        nodes.append({
            "node_id": node_id,
            "date": d,
            "cluster_idx": idx,
            "cluster_size": len(cl),
            "elements": ",".join(sorted(cl)),
        })
        node_index_map[(d, idx)] = node_id
        node_id += 1

df_nodes = pd.DataFrame(nodes)


# ---------------------------------------------------------------
# Construct the edge set E
# ---------------------------------------------------------------
for t in range(len(dates) - 1):
    d_i   = dates[t]
    d_next = dates[t + 1]

    C_i    = V_prime[d_i]
    C_next = V_prime[d_next]

    for j, S_ij in enumerate(C_i):
        set_ij = set(S_ij)

        for k, S_next in enumerate(C_next):
            set_next = set(S_next)
            inter = set_ij & set_next
            w = len(inter)

            if w > 0:
                edges.append({
                    "source": node_index_map[(d_i, j)],
                    "target": node_index_map[(d_next, k)],
                    "date_from": d_i,
                    "date_to": d_next,
                    "intersection_weight": w,
                    "shared_elements": ",".join(sorted(inter)),
                })

df_edges = pd.DataFrame(edges)


# ===============================================================
# SAVE OUTPUTS (to scripts/outputs)
# ===============================================================
nodes_path = os.path.join(OUTPUT_DIR, "temporal_nodes.csv")
edges_path = os.path.join(OUTPUT_DIR, "temporal_edges.csv")

df_nodes.to_csv(nodes_path, index=False)
print(f"\nSaved {nodes_path}")

df_edges.to_csv(edges_path, index=False)
print(f"Saved {edges_path}")

# ===============================================================
# Quick Statistics Summary (for sanity-check + analysis)
# ===============================================================
print("\n=== TEMPORAL GRAPH SUMMARY (TGC Statistics) ===")

num_nodes = len(df_nodes)
num_edges = len(df_edges)

print(f"Total nodes (|V'|): {num_nodes}")
print(f"Total edges (|E|): {num_edges}")

# cluster size stats
print("\nCluster Size Statistics (|S_{i,j}|):")
print(df_nodes["cluster_size"].describe())

# edge weight stats
if num_edges > 0:
    print("\nEdge Weight Statistics (|S_{i,j} ∩ S_{i+1,k}|):")
    print(df_edges["intersection_weight"].describe())
else:
    print("\nNo edges were created (no overlaps).")

# basic density
possible_edges = (len(dates) - 1)  # between each consecutive τ_i
avg_edges_per_step = num_edges / possible_edges if possible_edges > 0 else 0

print(f"\nAverage edges per day-step: {avg_edges_per_step:.2f}")

# dates
print(f"\nDate range: {dates[0]} → {dates[-1]}")
print("===============================================\n")

print("\nDONE — This implements Section 2.3 + Part II exactly.\n")

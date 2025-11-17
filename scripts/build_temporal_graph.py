import pickle
import networkx as nx

print("Loading Jaccard clusters...")

with open("outputs/jaccard_clusters.pkl", "rb") as f:
    cluster_data = pickle.load(f)

# Correct structure
windows = cluster_data["windows"]         # [10,30,50]
clusters_all = cluster_data["clusters"]   # dict: window → date → clusters list

print(f"Loaded windows: {windows}")
print("Building temporal cluster graph...\n")

G = nx.DiGraph()

cluster_ids = {}
node_counter = 0

# ---------------------------
# BUILD NODES
# ---------------------------
for window in windows:
    for date, cluster_list in clusters_all[window].items():

        for cluster in cluster_list:
            ctuple = tuple(sorted(cluster))

            node_id = f"{window}|{date}|{node_counter}"

            G.add_node(
                node_id,
                window=window,
                date=date,
                members=ctuple,
                size=len(ctuple),
            )

            cluster_ids[(window, date, ctuple)] = node_id
            node_counter += 1

print("Nodes built.")


# ---------------------------
# BUILD EDGES BASED ON OVERLAP
# ---------------------------
print("Adding temporal edges...")

for window in windows:
    dates = sorted(clusters_all[window].keys())

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev_clusters = clusters_all[window][prev_date]
        curr_clusters = clusters_all[window][curr_date]

        for prev_c in prev_clusters:
            prev_tuple = tuple(sorted(prev_c))

            for curr_c in curr_clusters:
                curr_tuple = tuple(sorted(curr_c))

                overlap = len(set(prev_tuple) & set(curr_tuple))

                # Only add edges if clusters meaningfully overlap
                if overlap >= 1:  
                    prev_id = cluster_ids[(window, prev_date, prev_tuple)]
                    curr_id = cluster_ids[(window, curr_date, curr_tuple)]
                    G.add_edge(prev_id, curr_id, weight=overlap)

print("Temporal edges added.")


# ---------------------------
# SAVE GRAPH
# ---------------------------
print("\nSaving temporal cluster graph...")

with open("outputs/temporal_cluster_graph.gpickle", "wb") as f:
    pickle.dump(G, f)

print("✓ Saved to outputs/temporal_cluster_graph.gpickle")
print(f"Total nodes: {len(G.nodes())}")
print(f"Total edges: {len(G.edges())}")
print("\nDone!")

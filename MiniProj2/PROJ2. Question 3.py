import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

#Problem 3
#1.
file_path = "/Users/jaydenjohnston/Library/CloudStorage/OneDrive-Personal/SPRING 2025/Data Mining/Mini Project 2/email-Eu-core.txt"

G = nx.read_edgelist(file_path, delimiter=' ', nodetype=int)

largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
sample_size = 100  
sampled_nodes = list(G_lcc.nodes())[:sample_size]
G_sample = G_lcc.subgraph(sampled_nodes)

plt.figure(figsize=(10, 8))
nx.draw(G_sample, node_size=20, edge_color="gray", with_labels=False)
plt.title("Visualization of Sampled Email Network (100 Nodes)")
plt.show()

#2.
closeness_centrality = nx.closeness_centrality(G_lcc)

top_10_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

df_closeness = pd.DataFrame(top_10_closeness, columns=["Node", "Closeness Centrality"])

print(df_closeness.to_string(index=False))
#3.

betweenness_centrality = nx.betweenness_centrality(G_lcc)

top_10_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

df_betweenness = pd.DataFrame(top_10_betweenness, columns=["Node", "Betweenness Centrality"])

print(df_betweenness.to_string(index=False))

#4
clustering_coefficients = nx.clustering(G_lcc)

top_10_clustering = sorted(clustering_coefficients.items(), key=lambda x: x[1], reverse=True)[:10]

df_clustering = pd.DataFrame(top_10_clustering, columns=["Node", "Clustering Coefficient"])

print(df_clustering.to_string(index=False))


#5
eigenvector_centrality = nx.eigenvector_centrality(G_lcc, max_iter=1000)

top_10_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

df_eigenvector = pd.DataFrame(top_10_eigenvector, columns=["Node", "Eigenvector Centrality"])

print(df_eigenvector.to_string(index=False))

#6.
pagerank_scores = nx.pagerank(G_lcc)

top_10_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]

df_pagerank = pd.DataFrame(top_10_pagerank, columns=["Node", "PageRank"])

print(df_pagerank.to_string(index=False))

#8
degree_sequence = [d for n, d in G_lcc.degree()]
degree_counts = np.bincount(degree_sequence)
nonzero_degrees = np.nonzero(degree_counts)[0]
probabilities = degree_counts[nonzero_degrees] / sum(degree_counts)

log_k = np.log10(nonzero_degrees)
log_f_k = np.log10(probabilities)

slope, intercept, r_value, p_value, std_err = linregress(log_k, log_f_k)

plt.figure(figsize=(8, 6))
plt.scatter(log_k, log_f_k, label="Data points", color="blue", alpha=0.6)
plt.plot(log_k, slope * log_k + intercept, color="red", linestyle="dashed", label="Least-squares fit")

plt.xlabel("log k (Degree)")
plt.ylabel("log f(k) (Probability)")
plt.title("Log-Log Plot of Degree Distribution")
plt.legend()
plt.show()

power_law = slope < 0 

power_law


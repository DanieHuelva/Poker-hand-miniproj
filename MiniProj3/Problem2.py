from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import sys
from sklearn.datasets import make_blobs
from collections import defaultdict
from enum import Enum



## Problem 2.1
def getKmeans(data, k, eps):
    num_points, num_features = data.shape
    indices = np.random.choice(num_points, size=k, replace=False)
    centroids = data[indices]
    while True:
        #Assign each point to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        #new centroids
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        #Check
        movement = np.linalg.norm(new_centroids - centroids)
        if movement < eps:
            break
        centroids = new_centroids
    return centroids, labels




#Problem 2.2
def getDbscan(data, minpts, eps):
    n = data.shape[0]
    labels = np.full(n, -1)  # All points start as unassigned (-1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    point_types = ['noise'] * n

    def region_query(point_idx):   #getting neighbors
        dists = np.linalg.norm(data - data[point_idx], axis=1)
        return np.where(dists <= eps)[0]

    for point_idx in range(n):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = region_query(point_idx)

        if len(neighbors) < minpts:             #if has less neighbor make it noise
            point_types[point_idx] = 'noise'
        else:
            # Core point
            labels[point_idx] = cluster_id
            point_types[point_idx] = 'core'    
            seeds = list(neighbors)

            i = 0
            while i < len(seeds):       #checking neighbors
                neighbor_idx = seeds[i]
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    new_neighbors = region_query(neighbor_idx)
                    if len(new_neighbors) >= minpts:
                        point_types[neighbor_idx] = 'core'
                        seeds += list(set(new_neighbors) - set(seeds))  # expand cluster
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
                    if point_types[neighbor_idx] != 'core':
                        point_types[neighbor_idx] = 'border'
                i += 1
            cluster_id += 1
    return labels, point_types





data = pd.read_csv('MiniProj3/customers.csv')

##Im just checking it here!
scalar = StandardScaler()
STD = scalar.fit_transform(data.to_numpy())


#Problem 2.1 Test case
# print(getKmeans(STD, 8, 0.5))

#Problem 2.2 Test case
# print(getDbscan(STD, 5, 0.5))
# print(getDbscan(STD, 3, 0.8))



#Problem 3.1
# pca = PCA(n_components=2)
# RPCA = pca.fit_transform(STD)
# kmeans = KMeans(n_clusters=6, random_state=80)
# clusters_kmeans = kmeans.fit_predict(RPCA)
# centroids = kmeans.cluster_centers_
# plt.figure(figsize=(6, 6))
# plt.scatter(RPCA[:, 0], RPCA[:, 1], c=clusters_kmeans, cmap='viridis', alpha=0.9, edgecolor='k')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label="Centroids")
# plt.xlabel("First Component")
# plt.ylabel("Second Component")
# plt.legend()
# plt.grid(True)
# plt.show()


#Problem 3.2
# pca = PCA()
# pca.fit(STD)
# ex_var = pca.explained_variance_ratio_
# cum_var = np.cumsum(ex_var)

# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
# plt.xlabel("Number of Principal Components (r)")
# plt.ylabel("Cumulative Variance Explained")
# plt.title("Explained Variance vs Number of Principal Components")
# plt.grid(True)
# plt.show()


#Problem 3.3
# pca = PCA(n_components=6)
# RPCA = pca.fit_transform(STD)

# k_values = list(range(2, 11))  # from 2 to 10
# inertia_original = []
# inertia_reduced = []

# # Original data
# for k in k_values:
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     km.fit(STD)
#     inertia_original.append(km.inertia_)

# # PCA-reduced data
# for k in k_values:
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     km.fit(RPCA)
#     inertia_reduced.append(km.inertia_)

# plt.figure(figsize=(12, 5))

# # Original data plot
# plt.subplot(1, 2, 1)
# plt.plot(k_values, inertia_original, marker='o')
# plt.title("K-Means on Original Data")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Objective (Inertia)")
# plt.grid(True)

# # Reduced data plot
# plt.subplot(1, 2, 2)
# plt.plot(k_values, inertia_reduced, marker='o', color='orange')
# plt.title("K-Means on PCA-Reduced Data")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Objective (Inertia)")
# plt.grid(True)

# plt.tight_layout()
# plt.show()



#Problem 3.4
pca = PCA(n_components=6)
RPCA = pca.fit_transform(STD)

# Parameters
minpts_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
eps_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

def dbscan_cluster_counts(data, eps_list, minpts_list):
    results = []
    for eps in eps_list:
        for minpts in minpts_list:
            model = DBSCAN(eps=eps, min_samples=minpts).fit(data)
            labels = model.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            results.append({
                'eps': eps,
                'minpts': minpts,
                'clusters': n_clusters
            })
    return pd.DataFrame(results)

# Run on both datasets
original_results = dbscan_cluster_counts(STD, eps_list, minpts_list)
pca_results = dbscan_cluster_counts(RPCA, eps_list, minpts_list)
print(original_results, pca_results)





### THIS IS TO CHECK DBSCAN DO NOT ERASE

# # Fit DBSCAN to your data
# db = DBSCAN(eps=0.8, min_samples=5).fit(STD)

# # Labels: cluster index (-1 means noise)
# labels = db.labels_

# # Indices of core points
# core_indices = db.core_sample_indices_

# # Create boolean mask for core points
# core_mask = np.zeros_like(labels, dtype=bool)
# core_mask[core_indices] = True

# # Create a list of point types (same length as labels)
# point_types = []

# for i in range(len(labels)):
#     if labels[i] == -1:
#         point_types.append('noise')
#     elif core_mask[i]:
#         point_types.append('core')
#     else:
#         point_types.append('border')

# print(labels, point_types)

# # Plot DBSCAN clustering results
# plt.figure(figsize=(6, 6))
# plt.scatter(RPCA[:, 0], RPCA[:, 1], c=clusters_dbscan, cmap='rainbow', alpha=0.5, edgecolor='k')
# plt.xlabel("First Component")
# plt.ylabel("Second Component")
# plt.grid(True)
# plt.show()



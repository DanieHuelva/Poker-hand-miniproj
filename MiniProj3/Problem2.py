from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import sys
from sklearn.datasets import make_blobs




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


def getDbscan(data, minpts, eps):
    n = data.shape[0]
    labels = np.full(n, -1)  # All points start as unassigned (-1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    point_types = ['noise'] * n

    def region_query(point_idx):
        dists = np.linalg.norm(data - data[point_idx], axis=1)
        return np.where(dists <= eps)[0]

    for point_idx in range(n):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = region_query(point_idx)

        if len(neighbors) < minpts:
            point_types[point_idx] = 'noise'
        else:
            # Core point
            labels[point_idx] = cluster_id
            point_types[point_idx] = 'core'
            seeds = list(neighbors)

            i = 0
            while i < len(seeds):
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




matrix_data = pd.read_csv('MiniProj3/customers.csv')
(getKmeans(matrix_data.to_numpy(), 8, sys.float_info.epsilon))

# Parameters for DBSCAN
eps = 0.8
minpts = 5

# Run your custom DBSCAN
custom_labels, custom_types = getDbscan(matrix_data.to_numpy(), minpts, eps)

# Run actual DBSCAN from scikit-learn
sklearn_dbscan = DBSCAN(eps=eps, min_samples=minpts)
sklearn_labels = sklearn_dbscan.fit_predict(matrix_data.to_numpy())

# Plotting the results to visually compare

plt.figure(figsize=(12, 6))

# Custom DBSCAN clusters
plt.subplot(1, 2, 1)
plt.title("Custom DBSCAN Clusters")
plt.scatter(matrix_data[:, 0], matrix_data[:, 1], c=custom_labels, cmap='viridis')
plt.colorbar(label='Cluster ID')

# Sklearn DBSCAN clusters
plt.subplot(1, 2, 2)
plt.title("Sklearn DBSCAN Clusters")
plt.scatter(matrix_data[:, 0], matrix_data[:, 1], c=sklearn_labels, cmap='viridis')
plt.colorbar(label='Cluster ID')

plt.tight_layout()
plt.show()

##Im just checking it here!
# scalar = StandardScaler()
# STD = scalar.fit_transform(matrix_data)
# pca = PCA(n_components=2)
# RPCA = pca.fit_transform(STD)
# kmeans = KMeans(n_clusters=8, random_state=42)
# clusters_kmeans = kmeans.fit_predict(matrix_data.to_numpy())
# centroids = kmeans.cluster_centers_
# plt.figure(figsize=(6, 6))
# plt.scatter(RPCA[:, 0], RPCA[:, 1], c=clusters_kmeans, cmap='viridis', alpha=0.9, edgecolor='k')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label="Centroids")
# plt.xlabel("First Component")
# plt.ylabel("Second Component")
# plt.legend()
# plt.grid(True)
# plt.show()
#part 1
from sklearn.metrics import precision_score
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clustering_precision(true_labels, pred_labels):
    """
    Calculates clustering precision by finding best 1-to-1 mapping between predicted and true labels.
    """
    labels_true = LabelEncoder().fit_transform(true_labels)
    labels_pred = LabelEncoder().fit_transform(pred_labels)
    
    n_clusters = len(np.unique(labels_true))
    n_classes = len(np.unique(labels_pred))
    contingency = np.zeros((n_clusters, n_classes))

    for i in range(len(labels_true)):
        contingency[labels_true[i], labels_pred[i]] += 1

    return np.sum(np.amax(contingency, axis=1)) / np.sum(contingency)
############################
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def getKmeans(data, k, eps):
    num_points, num_features = data.shape
    indices = np.random.choice(num_points, size=k, replace=False)
    centroids = data[indices]
    while True:
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        movement = np.linalg.norm(new_centroids - centroids)
        if movement < eps:
            break
        centroids = new_centroids
    return centroids, labels

def clustering_precision(true_labels, pred_labels):
    labels_true = LabelEncoder().fit_transform(true_labels)
    labels_pred = LabelEncoder().fit_transform(pred_labels)
    n_clusters = len(np.unique(labels_true))
    n_classes = len(np.unique(labels_pred))
    contingency = np.zeros((n_clusters, n_classes))
    for i in range(len(labels_true)):
        contingency[labels_true[i], labels_pred[i]] += 1
    return np.sum(np.amax(contingency, axis=1)) / np.sum(contingency)

# Load dataset
df = pd.read_csv("MiniProj3/customers.csv")
true_labels = df["Channel"]
df_numeric = df.drop(columns=["Channel", "Region"], errors='ignore')

# Standardize
scaler = StandardScaler()
STD = scaler.fit_transform(df_numeric)

# PCA
pca = PCA(n_components=6)
RPCA = pca.fit_transform(STD)

# Define k range
k_values = list(range(2, 8))
precision_k_original = []
precision_k_reduced = []

# Run getKmeans and compute precision
for k in k_values:
    _, labels_orig = getKmeans(STD, k, eps=0.01)
    _, labels_red = getKmeans(RPCA, k, eps=0.01)
    precision_k_original.append(clustering_precision(true_labels, labels_orig))
    precision_k_reduced.append(clustering_precision(true_labels, labels_red))

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(k_values, precision_k_original, marker='o')
axes[0].set_title("Precision vs k (Original Data)")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Precision")

axes[1].plot(k_values, precision_k_reduced, marker='o')
axes[1].set_title("Precision vs k (Reduced Data)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Precision")

plt.tight_layout()
plt.show()

###########################part 2
def getDbscan(data, minpts, eps):
    n = data.shape[0]
    labels = np.full(n, -1)  
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
                        seeds += list(set(new_neighbors) - set(seeds))
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
                    if point_types[neighbor_idx] != 'core':
                        point_types[neighbor_idx] = 'border'
                i += 1
            cluster_id += 1
    return labels, point_types



eps_values = np.arange(0.2, 1.4, 0.2)
precision_eps_original = []

for eps in eps_values:
    labels, _ = getDbscan(STD, minpts=5, eps=eps)
    if len(set(labels)) > 1 and -1 not in set(labels):
        precision = clustering_precision(true_labels, labels)
    else:
        precision = 0
    precision_eps_original.append(precision)

# fix epsilon and vary minpts
minpts_values = list(range(2, 12))
precision_minpts_original = []

for minpts in minpts_values:
    labels, _ = getDbscan(STD, minpts=minpts, eps=0.8)
    if len(set(labels)) > 1 and -1 not in set(labels):
        precision = clustering_precision(true_labels, labels)
    else:
        precision = 0
    precision_minpts_original.append(precision)

# plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(eps_values, precision_eps_original, marker='o', color='purple')
axes[0].set_title("Precision vs Epsilon (DBSCAN, Original Data)")
axes[0].set_xlabel("Epsilon")
axes[0].set_ylabel("Precision")

axes[1].plot(minpts_values, precision_minpts_original, marker='o', color='orange')
axes[1].set_title("Precision vs MinPts (DBSCAN, Original Data)")
axes[1].set_xlabel("MinPts")
axes[1].set_ylabel("Precision")

plt.tight_layout()
plt.show()

################################# part 3
# Run DBSCAN using custom getDbscan on PCA-reduced data for each (eps, minpts) pair
precision_by_eps_reduced = {eps: [] for eps in eps_values}
precision_by_minpts_reduced = {minpts: [] for minpts in minpts_values}

for eps in eps_values:
    for minpts in minpts_values:
        labels, _ = getDbscan(RPCA, minpts=minpts, eps=eps)
        if len(set(labels)) > 1 and -1 not in set(labels):
            precision = clustering_precision(true_labels, labels)
        else:
            precision = 0
        precision_by_eps_reduced[eps].append(precision)
        precision_by_minpts_reduced[minpts].append(precision)

# Compute average precision across settings for reduced data
avg_precision_eps_reduced = [np.mean(precision_by_eps_reduced[eps]) for eps in eps_values]
avg_precision_minpts_reduced = [np.mean(precision_by_minpts_reduced[minpts]) for minpts in minpts_values]

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(eps_values, avg_precision_eps_reduced, marker='o', color='blue')
axes[0].set_title("Avg Precision vs Epsilon (DBSCAN, PCA-Reduced Data)")
axes[0].set_xlabel("Epsilon")
axes[0].set_ylabel("Average Precision")

axes[1].plot(minpts_values, avg_precision_minpts_reduced, marker='o', color='green')
axes[1].set_title("Avg Precision vs MinPts (DBSCAN, PCA-Reduced Data)")
axes[1].set_xlabel("MinPts")
axes[1].set_ylabel("Average Precision")

plt.tight_layout()
plt.show()

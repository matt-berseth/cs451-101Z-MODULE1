# Import necessary libraries
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Generate a random dataset with 3 blobs (clusters)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Visualize the generated data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Generated Data")
plt.savefig("generated_data.png")

# Initialize and fit the KMeans clusterer
# We specify the number of clusters (n_clusters) based on our knowledge of the data. 
# In this case, we know we generated 3 clusters, but in real scenarios, this may require some analysis.
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Get cluster assignments for each data point
labels = kmeans.labels_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
centers = kmeans.cluster_centers_
print(f"Cluster centers:\n {centers}")

# TODO: what points below to what colors in the image?

plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.75)
plt.title("KMeans Clustering")
plt.savefig("unsupervised.png")

# If you"d like to predict the closest cluster for new data points:
# new_data = [[x1, x2], ...]
# cluster_assignments = kmeans.predict(new_data)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


x, y = make_blobs(n_samples=500, centers=5, cluster_std=0.6, random_state=40)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

cluster_numbers = [2,3,4,5,6,7,8,9]
inertia = []
silhouette_scores = []


for k in cluster_numbers:
    kmeans = KMeans(n_clusters=k, random_state=40).fit(x)
    inertia.append(kmeans.inertia_)
    
    silhouette_avg = silhouette_score(x, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    
# Inertia always goes down as you increse the number of clusters
print(inertia)

# This is the Elbow method
plt.plot(cluster_numbers, inertia, marker='o')
plt.show()

# There are massive decreases in inertia until 5
# So 5 is the optimal amount, which makes sense given the information above!!!


print(silhouette_scores)

plt.plot(cluster_numbers, silhouette_scores, marker='o')
plt.show()

# 5 is the peak, because the clusters get closer together after 5, which is not optimal
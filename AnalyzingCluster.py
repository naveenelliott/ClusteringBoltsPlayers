import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import seaborn as sns
from kneed import KneeLocator

total = pd.read_csv('ClusteringBoltsPlayers/FormattedDataKMeansNew.csv')
players = total['Name']
del total['Name'], total['Team Name'], total['Total Shots']

# Using domain knowledge to filter out
total.drop(columns={'Cross %', 'Long Pass %', 'Minutes', 'Goal Against'}, inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(total)
scaled_df = pd.DataFrame(scaled_data, columns=total.columns)

correlation_matrix = scaled_df.corr()
high_corr_pairs = []

# Iterate through the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.7 or correlation_matrix.iloc[i, j] < -0.7:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# Convert to a DataFrame for better readability
high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])

# Apply PCA
pca = PCA()  # Number of components can be adjusted
pca_data = pca.fit_transform(scaled_data)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Select the number of components that explain a desired amount of variance, e.g., 90%
n_components = np.argmax(cumulative_explained_variance >= 0.9) + 1

pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(scaled_data)

# Inspect principal component loadings
loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'PC{i+1}' for i in range(n_components)], 
                        index=total.columns)
# Set a threshold for significant loadings (absolute value)
threshold = 0.5

# Keep features with loadings > threshold in any principal component
significant_features = loadings[(loadings.abs() > threshold).any(axis=1)]

# Plot the loadings for the first principal component
plt.figure(figsize=(10, 6))
plt.bar(loadings.index, loadings['PC1'])
plt.xlabel('Features')
plt.ylabel('Loadings')
plt.title('Feature Loadings for First Principal Component')
plt.xticks(rotation=90)
plt.grid()
plt.show()

# not filtering anything yet, because I want to see what general looks like

# Silhouette Score
scaler = StandardScaler()
selected = scaler.fit_transform(scaled_df)


cluster_numbers = list(range(2, 26))
inertia = []
silhouette_scores = []


for k in cluster_numbers:
    kmeans = KMeans(n_clusters=k, random_state=40).fit(selected)
    inertia.append(kmeans.inertia_)
    
    silhouette_avg = silhouette_score(selected, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    


plt.plot(cluster_numbers, silhouette_scores, marker='o')
plt.xticks(cluster_numbers)
plt.grid(True)
plt.show()

# 8 and 14 with general


# Elbow Method
# Apply KMeans for different numbers of clusters

inertia = []
for k in range(1, 19):  # Test clusters from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data)  # Use PCA-reduced data or original data
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 19), inertia, marker='o')
plt.xticks(range(1,19))
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.grid()
plt.show()

knee_locator = KneeLocator(range(1, 19), inertia, curve="convex", direction="decreasing")
optimal_k = knee_locator.knee

# again 8 or 14
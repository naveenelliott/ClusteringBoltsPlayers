import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min

total = pd.read_csv('ClusteringBoltsPlayers/FormattedDataKMeansNew.csv')
players = total['Name']
del total['Name'], total['Team Name'], total['Total Shots']

# Using domain knowledge to filter out
total.drop(columns={'Cross %', 'Long Pass %', 'Minutes', 'Goal Against'}, inplace=True)

# Standardize the data (optional but recommended)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(total)

# Apply K-Means with k=8
kmeans = KMeans(n_clusters=7, random_state=42)  # Set a seed for reproducibility
kmeans.fit(data_scaled)

# Add the cluster labels to the original DataFrame
total['cluster'] = kmeans.labels_
total['Player'] = players

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.9)  # Retain 90% of variance
reduced_data = pca.fit_transform(data_scaled)  # Reduced dataset
explained_variance = pca.explained_variance_ratio_  # Variance explained by each component

# Create a DataFrame for the reduced data
reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
reduced_df['Player'] = players


# Calculate pairwise weighted Euclidean distances
def weighted_euclidean(point, other_points, weights):
    return np.sqrt(np.sum(weights * (other_points - point) ** 2, axis=1))

# Find the closest player for each player
closest_points = []
weights = explained_variance  # PCA component weights

for i in range(len(reduced_df)):
    specific_point = reduced_data[i]  # The current player
    other_points = np.delete(reduced_data, i, axis=0)  # Exclude the current player
    distances = weighted_euclidean(specific_point, other_points, weights)  # Compute distances
    closest_index = np.argmin(distances)  # Find the index of the closest point
    # Map the index back to the original DataFrame
    closest_points.append(players[closest_index if closest_index < i else closest_index + 1])

# Add the closest player to the DataFrame
reduced_df['Closest Player'] = closest_points

total = pd.merge(total, reduced_df[['Player', 'Closest Player', 'PC1', 'PC2']], on=['Player'], how='inner')

cluster_centers = kmeans.cluster_centers_

# Create a DataFrame for better readability
cluster_centers_df = pd.DataFrame(columns=['PC1', 'PC2'])

cluster_centers_df['PC1'] = pca.transform(kmeans.cluster_centers_)[:, 0]
cluster_centers_df['PC2'] = pca.transform(kmeans.cluster_centers_)[:, 1]
cluster_centers_df['Cluster'] = range(0, len(cluster_centers_df))

cluster_centers_df.to_csv('ClusteringBoltsPlayers/ClusterCentersData.csv', index=False)


top_closest_columns = []

stat_columns = total.columns.difference(['Player', 'Closest Player', 'cluster'])

for index, row in total.iterrows():
    current_player = row['Player']
    compared_player = row['Closest Player']
    total_players = [current_player, compared_player]
    # Filter rows for the two players
    temp_compare = total.loc[total['Player'].isin(total_players), stat_columns].reset_index(drop=True)
    
    # Calculate the absolute differences
    differences = temp_compare.iloc[0] - temp_compare.iloc[1]
    differences = differences.abs()
    
    # Get the top 3 smallest differences
    top_3_columns = differences.nsmallest(3).index.tolist()
    top_closest_columns.append(top_3_columns)
    
total['Closest Statistics'] = top_closest_columns

total.to_csv('ClusteringBoltsPlayers/EndKMeansClustering.csv', index=False)
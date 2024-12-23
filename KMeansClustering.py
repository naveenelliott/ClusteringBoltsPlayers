import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min

bolts = pd.read_csv('ClusteringBoltsPlayers/FormattedDataKMeans.csv')
bolts = bolts.fillna(0)
update_bolts = bolts.copy()
update_bolts.set_index('Player Full Name', inplace=True)
update_bolts = update_bolts.drop(columns=['Team Name'])
temp = update_bolts.copy()

scaler = StandardScaler()
update_bolts = scaler.fit_transform(update_bolts)


cluster_numbers = list(range(2, 26))
inertia = []
silhouette_scores = []


for k in cluster_numbers:
    kmeans = KMeans(n_clusters=k, random_state=40).fit(update_bolts)
    inertia.append(kmeans.inertia_)
    
    silhouette_avg = silhouette_score(update_bolts, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    

# This is the Elbow method
plt.plot(cluster_numbers, inertia, marker='o')
plt.xticks(cluster_numbers)
plt.show()

# There are massive decreases in inertia until 10
# So 10 is the optimal amount, which makes sense given the information above!!!

plt.plot(cluster_numbers, silhouette_scores, marker='o')
plt.xticks(cluster_numbers)
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=9, random_state=40).fit(update_bolts)

# Perform PCA with 5 components
pca = PCA(n_components=5)
pca.fit(update_bolts)
explained_variance = pca.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), explained_variance, marker='o', linestyle='--')
plt.xticks(range(1, 6))
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# It looks like one or two pca components may be the best to explain the variance of the dataset
# First has 41.53% and Second has 13.51%


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(update_bolts)
explained_variance = pca.explained_variance_ratio_
reduced_df = pd.DataFrame(reduced_data, index=bolts['Player Full Name'], columns=['PC1', 'PC2'])
labels = kmeans.labels_
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], 
            s=300, c='red', marker='x')  # Cluster centers
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering (PCA-reduced Data)')
plt.show()

bolts['Cluster'] = labels

closest_points = []

# Calculate the weighted Euclidean distance for each point
weights = explained_variance

for i in range(len(reduced_df)):
    specific_point = reduced_df.iloc[i].values.reshape(1, -1)
    temp_df = reduced_df.drop(reduced_df.index[i])
    
    distances = np.sqrt(np.sum(weights * (temp_df.values - specific_point) ** 2, axis=1))
    closest_index = np.argmin(distances)
    closest_points.append(temp_df.index[closest_index])


# Add the closest point index to the DataFrame
bolts['Closest Point'] = closest_points

want = bolts[['Player Full Name', 'Closest Point']]

cluster_centers = kmeans.cluster_centers_

# Create a DataFrame for better readability
cluster_centers_df = pd.DataFrame(columns=['PC1', 'PC2'])

cluster_centers_df['PC1'] = pca.transform(kmeans.cluster_centers_)[:, 0]
cluster_centers_df['PC2'] = pca.transform(kmeans.cluster_centers_)[:, 1]
cluster_centers_df['Cluster'] = range(0, len(cluster_centers_df))

cluster_centers_df.to_csv('Streamlit/ClusterCentersData.csv', index=False)

temp.reset_index(inplace=True)
names = temp['Player Full Name']
columns_to_convert = temp.columns.difference(['Player Full Name'])


update_bolts = pd.DataFrame(update_bolts, columns=columns_to_convert)
update_bolts['Player Full Name'] = names
update_bolts['Closest Point'] = closest_points

update_bolts.rename(columns={'Assist': 'Assists', 'Blocked Cross': 'Blocked Crosses', 'Blocked Shot': 'Blocked Shots', 
                     'Dribble': 'Dribbles', 'Efforts on Goal': 'Shots', 'Goal': 'Goals', 'Loss of Poss': 'Possession Lost', 
                     'Pass Completion ': 'Pass %', 'Pass into Oppo Box': 'Passes into 18', 'Progr Pass Completion ': 'Forward Pass %', 
                     'Progr Regain ': 'Progressive Regain %', 'Total Att Aerials': 'Attacking Aerials', 
                     'Total Clears': 'Clearances', 'Total Def Aerials': 'Defensive Aerials', 
                     'Total Forward': 'Forward Passes', 'Total Interceptions': 'Interceptions', 'Total Long': 'Long Passes', 
                     'Total Pass': 'Passes', 'Total Recoveries': 'Recoveries', 'Total Tackles': 'Tackles'},
                    inplace=True)

top_closest_columns = []

for index, row in update_bolts.iterrows():
    current_player = row['Player Full Name']
    compared_player = row['Closest Point']
    total_players = [current_player, compared_player]
    temp_compare = update_bolts.loc[update_bolts['Player Full Name'].isin(total_players)].reset_index()
    temp_compare.drop(columns=['Player Full Name', 'Closest Point'], inplace=True)
    differences = temp_compare.iloc[0, 2:] - temp_compare.iloc[1, 2:]  # Exclude 'Player Full Name' and 'Closest Point'
    differences = differences.abs()
    differences = differences.loc[differences != 0]
    
    top_3_columns = differences.nsmallest(3).index.tolist()
    
    top_closest_columns.append(top_3_columns)
    
bolts['Closest Statistics'] = top_closest_columns

bolts.to_csv('ClusteringBoltsPlayers/EndKMeansClustering.csv', index=False)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler

bolts = pd.read_csv('ClusteringBoltsPlayers/FormattedDataKMeans.csv')
bolts = bolts.fillna(0)
update_bolts = bolts.copy()
update_bolts.set_index('Player Full Name', inplace=True)
update_bolts = update_bolts.drop(columns=['Team Name'])

scaler = StandardScaler()
update_bolts = scaler.fit_transform(update_bolts)

kmeans = KMeans(n_clusters=9, random_state=40).fit(update_bolts)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(update_bolts)
labels = kmeans.labels_

bolts['Cluster'] = labels

cluster_centers = kmeans.cluster_centers_

names = bolts['Player Full Name']
clusters = bolts['Cluster']
columns_to_convert = bolts.columns.difference(['Player Full Name', 'Team Name', 'Cluster'])
update_bolts = pd.DataFrame(update_bolts, columns=columns_to_convert)
update_bolts['Player Full Name'] = names
update_bolts['Cluster'] = clusters
update_bolts[['PC1', 'PC2']] = reduced_data
update_bolts['Color'] = None

update_bolts.to_csv('ClusteringBoltsPlayers/PCAPlayers.csv')
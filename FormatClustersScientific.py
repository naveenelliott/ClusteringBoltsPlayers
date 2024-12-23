import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import seaborn as sns

thirteens = pd.read_csv('Bolts Post-Match/BoltsThirteenGames/NinetyEightGame.csv')
fourteens = pd.read_csv('Bolts Post-Match/BoltsFourteenGames/NinetyEightGame.csv')
fifteens = pd.read_csv('Bolts Post-Match/BoltsFifteenGames/NinetyEightGame.csv')
sixteens = pd.read_csv('Bolts Post-Match/BoltsSixteenGames/NinetyNineGame.csv')
seventeens = pd.read_csv('Bolts Post-Match/BoltsSeventeenGames/NinetyEightGame.csv')
nineteens = pd.read_csv('Bolts Post-Match/BoltsNineteenGames/NinetyEightGame.csv')


def getFinalGrade(game_df):
    game_df.columns = game_df.iloc[3]
    game_df = game_df.iloc[4:]
    game_df = game_df.reset_index(drop=True)

    start_index = game_df.index[game_df["Period Name"] == "Running By Player"][0]

    # Find the index where "period name" is equal to "running by position player"
    end_index = game_df.index[game_df["Period Name"] == "Running By Position Player"][0]

# Select the rows between the two indices
    selected_rows = game_df.iloc[start_index:end_index]
    selected = selected_rows.reset_index(drop=True)
    selected = selected.iloc[1:]    
    return selected

thirteens = getFinalGrade(thirteens)
fourteens = getFinalGrade(fourteens)
fifteens = getFinalGrade(fifteens)
sixteens = getFinalGrade(sixteens)
seventeens = getFinalGrade(seventeens)
nineteens = getFinalGrade(nineteens)

total = pd.concat([thirteens, fourteens, fifteens, sixteens, seventeens, nineteens], ignore_index=True)

columns_we_want = ['Player Full Name', 'mins played', 'Position Tag', 'Team Name', 'Goal', 'Assist', 'Dribble', 'Offside', 'Stand. Tackle', 
 'Unsucc Stand. Tackle', 'Tackle', 'Unsucc Tackle', 'Def Aerial', 'Unsucc Def Aerial', 'Clear', 'Headed Clear',
 'Own Box Clear', 'Progr Rec', 'Unprogr Rec', 'Progr Inter', 'Unprogr Inter', 'Progr Regain ', 'Blocked Shot', 
 'Blocked Cross', 'Stand. Tackle Success ', 'Tackle Success ', 'Att 1v1', 'Att Aerial', 
 'Efforts on Goal', 'Header on Target', 'Header off Target', 'Shot on Target', 'Shot off Target', 'Att Shot Blockd', 
 'Unsucc Cross', 'Cross', 'Efficiency ', 'Side Back', 'Unsucc Side Back', 'Long', 'Unsucc Long', 'Forward', 
 'Unsucc Forward', 'Line Break', 'Pass into Oppo Box', 'Loss of Poss', 'Success', 'Unsuccess', 'Pass Completion ', 
 'Progr Pass Attempt ', 'Progr Pass Completion ', 'Foul Won', 'Foul Conceded', 'FK', 'Unsucc FK', 'Corner Kick', 
 'Short Corner', 'Unsucc Corner Kick']

total = total[columns_we_want]

total = total.loc[total['Position Tag'] != 'GK']

total.drop(columns=['Position Tag'], inplace=True)
columns_to_convert = total.columns.difference(['Player Full Name', 'Team Name'])
total[columns_to_convert] = total[columns_to_convert].apply(pd.to_numeric)
total.fillna(0, inplace=True)

total['Total Tackles'] = total['Stand. Tackle'] + total['Unsucc Stand. Tackle'] + total['Tackle'] + total['Unsucc Tackle']
total['Tackle %'] = ((total['Stand. Tackle'] + total['Tackle'])/total['Total Tackles'])*100
del total['Unsucc Stand. Tackle']
del total['Tackle'], total['Unsucc Tackle']

total['Total Def Aerials'] = total['Def Aerial'] + total['Unsucc Def Aerial']
del total['Def Aerial'], total['Unsucc Def Aerial']

total['Total Clears'] = total['Clear'] + total['Own Box Clear']
del total['Clear'], total['Headed Clear'], total['Own Box Clear']

total['Total Att Aerials'] = total['Header on Target'] + total['Header off Target'] + total['Att Aerial']
del total['Header on Target'], total['Header off Target']
del total['Shot off Target']
total['Total Crosses'] = total['Unsucc Cross'] + total['Cross']
del total['Unsucc Cross'], total['Cross']
total['Total Side Backs'] = total['Unsucc Side Back'] + total['Side Back']
total['Side Back %'] = (total['Side Back']/total['Total Side Backs']) * 100
del total['Unsucc Side Back'], total['Side Back']
total['Total Long'] = total['Unsucc Long'] + total['Long']
total['Long %'] = (total['Long']/total['Total Long']) * 100
del total['Unsucc Long'], total['Long']
total['Total Forward'] = total['Forward'] + total['Unsucc Forward']
del total['Forward'], total['Unsucc Forward']
total['Total Pass'] = total['Unsuccess'] + total['Success']
del total['Unsuccess'], total['Success']
total['Total FK'] = total['FK'] + total['Unsucc FK']
del total['FK'], total['Unsucc FK']
total['Total Corners'] = total['Corner Kick'] + total['Short Corner'] + total['Unsucc Corner Kick']
del total['Corner Kick'], total['Short Corner'], total['Unsucc Corner Kick']
# Could delete blocks next or combine them
total['Total Recoveries'] = total['Progr Rec'] + total['Unprogr Rec']
total['Total Interceptions'] = total['Progr Inter'] + total['Unprogr Inter']
del total['Progr Rec'], total['Unprogr Rec'], total['Progr Inter'], total['Unprogr Inter']
#del total['Blocked Shot'], total['Blocked Cross']

per_90 = ['Goal', 'Assist', 'Dribble', 'Blocked Shot', 'Blocked Cross',
       'Offside', 'Stand. Tackle', 'Total Recoveries', 'Total Interceptions', 
       'Efforts on Goal', 'Shot on Target', 'Total Def Aerials', 'Total Att Aerials',
       'Pass into Oppo Box', 'Loss of Poss', 'Total Tackles',
       'Total Clears', 'Total Long', 'Total Side Backs',
       'Total Crosses', 'Shot on Target', 'Progr Pass Attempt ', 'Total FK', 'Total Corners',
       'Total Forward', 'Total Pass', 'Line Break', 'Att 1v1', 'Foul Won', 'Foul Conceded']


for column in per_90:
    total[column] = (total[column]/total['mins played']) * 90
    
total = total.loc[total['mins played'] > 300]
del total['Att Aerial'], total['mins played']
total.drop(columns=['Stand. Tackle', 'Stand. Tackle Success ', 'Tackle Success ', 
                    'Att Shot Blockd'], inplace=True)

total = total.loc[total['Player Full Name'] != 'Quinn Pappendick']
goalkeepers = ['Aaron Choi', 'Jack Seaborn', 'Ben Marro', 'Casey Powers']

total = total.loc[~total['Player Full Name'].isin(goalkeepers)]
total.fillna(0, inplace=True)
players = total['Player Full Name']
del total['Player Full Name'], total['Team Name']

correlation_matrix = total.corr()
high_corr_pairs = []

# Iterate through the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.7 or correlation_matrix.iloc[i, j] < -0.7:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# Convert to a DataFrame for better readability
high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])

# Deleting variables with strong correlations together and the variable that contributes less to the explained variance
del total['Shot on Target'], total['Att 1v1'], total['Blocked Shot'], total['Total Pass'], total['Total Long']
del total['Side Back %'], total['Total Recoveries']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(total)

# Apply PCA
pca = PCA()  # Number of components can be adjusted
pca_data = pca.fit_transform(scaled_data)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid()
plt.show()

# Select the number of components that explain a desired amount of variance, e.g., 90%
n_components = np.argmax(cumulative_explained_variance >= 0.9) + 1

# Inspect principal component loadings
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=total.columns)

# Plot the loadings for the first principal component
plt.figure(figsize=(10, 6))
plt.bar(loadings.index, loadings['PC1'])
plt.xlabel('Features')
plt.ylabel('Loadings')
plt.title('Feature Loadings for First Principal Component')
plt.xticks(rotation=90)
plt.grid()
plt.show()

# Top features contributing to the first principal component
top_features_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(13)

selected = total[top_features_pc1.index]
selected_unscaled = selected.copy()

scaler = StandardScaler()
selected = scaler.fit_transform(selected)


cluster_numbers = list(range(2, 26))
inertia = []
silhouette_scores = []


for k in cluster_numbers:
    kmeans = KMeans(n_clusters=k, random_state=40).fit(selected)
    inertia.append(kmeans.inertia_)
    
    silhouette_avg = silhouette_score(selected, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    

# This is the Elbow method
plt.plot(cluster_numbers, inertia, marker='o')
plt.xticks(cluster_numbers)
plt.show()

plt.plot(cluster_numbers, silhouette_scores, marker='o')
plt.xticks(cluster_numbers)
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=10, random_state=40).fit(selected)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(selected)
labels = kmeans.labels_
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], 
            s=300, c='red', marker='x')  # Cluster centers
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering (PCA-reduced Data)')
plt.show()

selected_unscaled['Player Full Name'] = players
selected_unscaled['Cluster'] = labels
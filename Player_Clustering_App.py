import pandas as pd
import streamlit as st
import ast
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title='Player Clustering Calculator')

st.sidebar.success('Select a page above.')

st.title("Player Clustering Calculator")

st.markdown("Select The Player to See their Position Group and Closest Player in the Club")


clustered = pd.read_csv('EndKMeansClustering.csv')
clustered.sort_values('Player', ascending=True, inplace=True)

update_bolts = clustered.copy()

clustered_copy = clustered.copy()

def convert_to_list(cell):
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return cell

# Apply the function to the 'Closest Statistics' column
clustered['Closest Statistics'] = clustered['Closest Statistics'].apply(convert_to_list)

cluster_mapping = {
    0: 'Target CF',
    1: 'Build Up CB',
    2: 'Defensive Fullback',
    3: 'Wide Forward',
    4: 'Engine',
    5: 'Complete Winger',
    6: 'Creator Advanced Midfielder',
    7: 'Destroyer',
    8: 'Deep Lying Midfielder'
}

cluster_mapping2 = {
    0: 'These are the center forwards of the club who frequently score goals and take shots. These players do not win the ball back very much, but when they do, they do a good job of making progressive actions, like a pass to a teammate. They are not involved in the buildup much, but are good in the air.',
    1: 'These are center backs of the club who are extremely involved in the buildup with high completion percentages and a large number of progressive passes. Additionally, they are very involved defensively with blocked shots and crosses, defensive aerials, and clearances. ',
    2: 'These fullbacks don’t appear to get forward much and contribute to the ball progression of the team; however, they frequently block crosses and are solid in the air. They do a good job of intercepting and recovering the ball as well.',
    3: 'These are wingers who are frequently involved in goals and assists for the club. They often take on opponents in 1v1 situations and while they are successful in beating them, they do lose possession often. They take many shots and aren’t often involved in the buildup, it is more about the end product (final dribble or pass) for these players.',
    4: 'These players are not the tidiest players in possession as their pass completion percentage is low and they occasionally lose possession. They regain possession fairly often though with above-average tackles, interceptions, and recoveries. ',
    5: 'These players are involved in goals and assists and take shots on, but not as much as the Wide Forwards in the club. They are more involved in the buildup with more (forward) passes and passes into the opposition box, but are not involved much defensively, with low blocked shots, crosses, and clearances.',
    6: 'Despite taking a good amount of shots, these types of players do not record many goals; however, they have many assists and excel at dribbling. They are excellent at progressing the ball into the opposition box and do a good job of regaining possession and making a positive action after (like a pass to a teammate). They are involved in the buildup of the team and will try to cross the ball into the box on occasion. ',
    7: 'These players are not involved at all in the end product of the team and are not the best in the buildup; however, they have very strong defensive numbers. They will tackle opponents and recover and intercept the ball at a rate higher than any other cluster in the club. They also are good in the air and frequently clear the ball. While these players frequently record defensive statistics, they are not the most efficient, with low progressive regain percentages and a low tackle success rate.',
    8: 'These players are not involved in the final product much, but they do contribute to the buildup through their forward passes and avoid losing possession. They don’t record many tackles, but when they recover or intercept the ball, there is a positive action after. '
}

# Replace cluster numbers with corresponding strings
clustered['Other Cluster'] = clustered['cluster'].replace(cluster_mapping)

clustered['Cluster Description'] = clustered['cluster'].replace(cluster_mapping2)

players = list(clustered['Player'].unique())

# Initialize prev_player in session state if not already present
if "prev_player" not in st.session_state:
    st.session_state["prev_player"] = players[0]  # Default to the first player if no previous selection

# Select player with the previous player as the default value
selected_player = st.selectbox('Choose the Bolts player:', players, index=players.index(st.session_state["prev_player"]))

clustered_player = clustered.loc[clustered['Player'] == selected_player]


position_group = clustered_player['Other Cluster'].values
position_group_desc = clustered_player['Cluster Description'].values
closest_player = clustered_player['Closest Player'].values

closest_player_stats = clustered_player['Closest Statistics'].reset_index(drop=True)

st.write(f"{selected_player}'s position group is {position_group[0]}. {position_group_desc[0]}")

st.write(f"{selected_player}'s closest comparable player is {closest_player[0]}.")

st.write(f"{selected_player}'s closest statistics with {closest_player[0]} are {closest_player_stats[0][0]}, {closest_player_stats[0][1]}, and {closest_player_stats[0][2]}.")

center_df = pd.read_csv('ClusterCentersData.csv')


cluster_highlight = update_bolts.loc[update_bolts['Player'] == selected_player]
selected_cluster = cluster_highlight['cluster'].values[0]

fig = go.Figure()

other_clusters_df = update_bolts.loc[update_bolts['cluster'] != selected_cluster]
fig.add_trace(
    go.Scatter(
        mode='markers',
        x=other_clusters_df['PC1'],
        y=other_clusters_df['PC2'],
        marker=dict(
            color='gray',
            size=10,
        ),
        name='Other Clusters',
        text=other_clusters_df['Player'],  # Use 'text' instead of 'hoverinfo'
        hoverinfo='text', 
        showlegend=True
    )
)

selected_cluster_df = update_bolts.loc[update_bolts['cluster'] == selected_cluster]
cluster_pnames = selected_cluster_df['Player']
fig.add_trace(
    go.Scatter(
        mode='markers',
        x=selected_cluster_df['PC1'],
        y=selected_cluster_df['PC2'],
        marker=dict(
            color='lightblue',
            size=10,
        ),
        name=f"{selected_player}'s Cluster",
        text=selected_cluster_df['Player'],  # Use 'text' instead of 'hoverinfo'
        hoverinfo='text', 
        showlegend=True
    )
)


selected_player_trace = update_bolts[update_bolts['Player'] == selected_player]
fig.add_trace(
    go.Scatter(
        mode='markers',
        x=selected_player_trace['PC1'],
        y=selected_player_trace['PC2'],
        marker=dict(
            color='blue',
            size=12,
        ),
        name=selected_player,
        hoverinfo='name',
        showlegend=True
    )
)

center_df = center_df.loc[center_df['Cluster'] == selected_cluster]
fig.add_trace(
    go.Scatter(
        mode='markers',
        x=center_df['PC1'],
        y=center_df['PC2'],
        marker=dict(
            color='red',
            size=10,
            symbol='x',
        ),
        name="Cluster Center",
        hoverinfo='name',
        showlegend=True
    )
)


# Update layout properties
fig.update_layout(
    title="Boston Bolts Player Style Clusters",
    title_font_size=20,  # Adjust the font size as needed
    title_x=0.2,  # Center the title horizontally
    title_y=0.85,
    xaxis=dict(
        title='',
        showticklabels=False  # Hide x-axis tick labels
    ),
    yaxis=dict(
        title='',
        showticklabels=False  # Hide y-axis tick labels
    ),
    hovermode='closest',  # Show closest data on hover
    showlegend=True,
    annotations=[
        dict(
            text=f"(With {selected_player}'s Cluster in Blue)",
            xref="paper",  # Position relative to paper
            yref="paper",  # Position relative to paper
            x=0.5,         # X position (0 to 1)
            y=1.0,        # Y position (0 to 1)
            showarrow=False,
            font=dict(size=12),
            bordercolor='blue'
        )
    ]
)


# Show the plot
st.plotly_chart(fig)


cols_we_want = ['Player', 'Goal', 'Assist', 'Dribble',
       'Progr Regain ', 'Blocked Shot', 'Blocked Cross', 'Efforts on Goal',
       'Pass into Oppo Box', 'Loss of Poss', 'Pass Completion ',
       'Progr Pass Completion ', 'Total Tackles', 'Tackle %',
       'Total Def Aerials', 'Total Clears', 'Total Att Aerials',
       'Total Crosses', 'Total Long', 'Total Forward', 'Total Pass',
       'Total Recoveries', 'Total Interceptions']

clustered_copy = clustered_copy[cols_we_want]

def calculate_percentiles(df):
    percentiles_df = pd.DataFrame(index=df.index, columns=df.columns)

    saved_series = df['Player']
    df.drop(columns=['Player'], inplace=True)
    for player in df.index:
        for col in df.columns:
            # Calculate percentile of player in current column
            percentile = stats.percentileofscore(df[col], df.loc[player, col])
            percentiles_df.loc[player, col] = percentile
    percentiles_df['Player'] = saved_series
    return percentiles_df

# Calculate percentiles for every player in the DataFrame
clustered_copy = calculate_percentiles(clustered_copy)

player_desire = [selected_player, closest_player[0]]
#clustered_copy = clustered_copy.loc[(clustered_copy['Player'].isin(player_desire))]
#clustered_copy = clustered_copy.set_index('Player').loc[player_desire].reset_index()

compared_player_df = clustered_copy.loc[clustered_copy['Player'] == closest_player[0]]
compared_player_df['Order'] = 1

clustered_copy = clustered_copy.loc[(clustered_copy['Player'].isin(cluster_pnames))]
clustered_copy = clustered_copy.set_index('Player').loc[cluster_pnames].reset_index()
clustered_copy['Order'] = 0

# Create boolean mask
mask = clustered_copy['Player'] == selected_player
mask2 = clustered_copy['Player'] == closest_player[0]

# Use .loc to modify 'Order' column where mask is True
clustered_copy.loc[mask, 'Order'] = 2
if not clustered_copy.loc[mask2].empty:
    clustered_copy.loc[mask2, 'Order'] = 1
else:
    clustered_copy = pd.concat([clustered_copy, compared_player_df], ignore_index=True)



st.session_state.clustered_copy = clustered_copy

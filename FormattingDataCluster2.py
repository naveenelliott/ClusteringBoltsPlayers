import mysql.connector
import pandas as pd
import glob
import os

# Database connection
connection = mysql.connector.connect(
    host="bostonbolts.cviw8wc8czxn.us-east-2.rds.amazonaws.com",
    user="bostonbolts",
    password="Naveen2!",
    database="bostonbolts_db",
    port=3306
)

if connection.is_connected():
    print("Successfully connected to the database")

query = "SELECT * FROM player_non_position_season_report;"
df = pd.read_sql(query, connection)


query = "SELECT * FROM player_non_position_game_report;"
df_game = pd.read_sql_query(query, connection)
df_game.columns = df_game.columns.str.replace('_', ' ', regex=True)
df_game = df_game[['Name', 'Team Name', 'Match Date', 'Minutes']]
df_game['Name'] = df_game['Name'].str.lower()

    
df.columns = df.columns.str.replace('_', ' ', regex=True)   
    
change_to_p90 = ['Goal', 'Assist', 'Shot on Target']

df['minutes per 90'] = df['Minutes']/90

df[change_to_p90] = df[change_to_p90].div(df['minutes per 90'], axis=0)

df.drop(columns=['minutes per 90'], inplace=True)


df = df.drop_duplicates(subset=['Name', 'Team Name'])

df_game = df_game.drop_duplicates(subset=['Name', 'Team Name', 'Match Date'])

df['Name'] = df['Name'].str.lower()

# Path to the folder containing CSV files
folder_path = 'Detailed_Match_Sessions/'

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

dataframes = []
# Filter filenames that contain both player_name and opp_name
for f in csv_files:
    file_path = os.path.join(folder_path, f)
    
    # Read the CSV file into a DataFrame
    pd_df = pd.read_csv(file_path)
    
    pd_df['athlete_name'] = pd_df['athlete_name'].str.lower()
    
    # Append the DataFrame to the list
    dataframes.append(pd_df)

    
playerdata_df = pd.concat(dataframes, ignore_index=True)

# Convert start_time from string to datetime in UTC
playerdata_df['start_time'] = pd.to_datetime(playerdata_df['start_time'])

# Set the timezone to UTC, then convert to EST
playerdata_df['start_time'] = playerdata_df['start_time'].dt.tz_convert('America/New_York')


playerdata_df['Match Date'] = pd.to_datetime(playerdata_df['start_time']).dt.strftime('%m/%d/%Y')

def rearrange_team_name(team_name):
    # Define age groups and leagues
    age_groups = ['U15', 'U16', 'U17', 'U19', 'U13', 'U14']
    leagues = ['MLS Next', 'NAL Boston', 'NAL South Shore']
    
    # Find age group in the team name
    for age in age_groups:
        if age in team_name:
            # Find the league part
            league_part = next((league for league in leagues if league in team_name), '')
            if league_part == 'NAL Boston':
                league_part = 'NALB'
            
            # Extract the rest of the team name
            rest_of_name = team_name.replace(age, '').replace('NAL Boston', '').replace(league_part, '').strip()
            
            
            # Construct the new team name
            return f"{rest_of_name} {age} {league_part}"
    
    # Return the original team name if no age group is found
    return team_name

# Apply the function to the 'team_name' column
playerdata_df['bolts team'] = playerdata_df['bolts team'].apply(rearrange_team_name)

del playerdata_df['metres_per_minute']

playerdata_df = playerdata_df[['bolts team', 'athlete_name', 'total_distance_m', 'total_high_intensity_distance_m',
       'high_intensity_events', 'total_sprint_distance_m', 'sprint_events',
       'acceleration_events', 'deceleration_events', 'max_speed_kph',
       'workload', 'workload_volume', 'workload_intensity', 'Match Date']]

end_phys = pd.merge(df_game, playerdata_df, left_on=['Team Name', 'Name', 'Match Date'], 
                    right_on=['bolts team', 'athlete_name', 'Match Date'], how='inner')

end_phys.drop(columns=['bolts team', 'athlete_name', 'Match Date'], inplace=True)

end_phys = end_phys.loc[end_phys['total_distance_m'] > 400]

end_phys = end_phys.loc[:, ~end_phys.columns.str.contains('workload', case=False)]



aggregation = {col: 'sum' for col in end_phys.columns if col not in ['max_speed_kph', 'Team Name', 'Name']}
aggregation['max_speed_kph'] = 'max'

# Apply the groupby with the custom aggregation
end_phys = end_phys.groupby(['Name', 'Team Name']).agg(aggregation).reset_index()

columns_to_adjust = end_phys.select_dtypes(exclude=['object', 'string']).columns.difference(['Minutes', 'max_speed_kph'])
end_phys[columns_to_adjust] = end_phys[columns_to_adjust].div(end_phys['Minutes'], axis=0).mul(90)

end_phys = end_phys.loc[end_phys['Minutes'] > 140]

end_phys = end_phys.sort_values('total_distance_m', ascending=False).reset_index(drop=True)

del end_phys['Minutes']

all_df = pd.merge(df, end_phys, on=['Name', 'Team Name'])

all_df.drop(columns=['ID', 'Yellow Card', 'Red Card', 'PK Missed', 'PK Scored'], inplace=True)

gks = ['ben marro', 'aaron choi', 'jack susi', 'casey powers', 'griffin taylor', 'daniel senichev', 'torran archer',
       'sy perkins', 'dylan jacobson', 'milo ketnouvong', 'jack seaborn']

all_df = all_df.loc[~all_df['Name'].isin(gks)]

all_df['Total Shots'] = all_df['Shot on Target'] + all_df['Shot off Target'] + all_df['Att Shot Blockd']

    
all_df = all_df.loc[all_df['Minutes'] > 270]
all_df.drop(columns=['Att Shot Blockd', 'Att 1v1', 'Shot on Target',
                     'Shot off Target', 'Crosses Claimed', 
                     'Save % ', 'Total Saves'], inplace=True)

connection.close()

all_df.to_csv('ClusteringBoltsPlayers/FormattedDataKMeansNew.csv', index=False)
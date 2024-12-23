import pandas as pd

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

total['Total Clears'] = total['Clear'] + total['Headed Clear'] + total['Own Box Clear']
del total['Clear'], total['Headed Clear'], total['Own Box Clear']

total['Total Att Aerials'] = total['Header on Target'] + total['Header off Target'] + total['Att Aerial']
del total['Header on Target'], total['Header off Target']
#total['Total Shots'] = total['Shot on Target'] + total['Shot off Target']
del total['Shot off Target']
total['Total Crosses'] = total['Unsucc Cross'] + total['Cross']
del total['Unsucc Cross'], total['Cross']
#total['Total Side Backs'] = total['Unsucc Side Back'] + total['Side Back']
#total['Side Back %'] = (total['Side Back']/total['Total Side Backs']) * 100
del total['Unsucc Side Back'], total['Side Back']
total['Total Long'] = total['Unsucc Long'] + total['Long']
#total['Long %'] = (total['Long']/total['Total Long']) * 100
del total['Unsucc Long'], total['Long']
total['Total Forward'] = total['Forward'] + total['Unsucc Forward']
del total['Forward'], total['Unsucc Forward']
total['Total Pass'] = total['Unsuccess'] + total['Success']
del total['Unsuccess'], total['Success']
del total['FK'], total['Unsucc FK']
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
       'Total Clears', 'Total Long',
       'Total Crosses', 'Shot on Target',
       'Total Forward', 'Total Pass']

for column in per_90:
    total[column] = (total[column]/total['mins played']) * 90
    
total = total.loc[total['mins played'] > 300]
del total['Att Aerial'], total['Progr Pass Attempt '], total['mins played']
total.drop(columns=['Stand. Tackle', 'Stand. Tackle Success ', 'Tackle Success ', 
                    'Att Shot Blockd', 'Efficiency ', 'Att 1v1', 'Shot on Target',
                    'Foul Won', 'Foul Conceded', 'Line Break', 'Offside'], inplace=True)

total = total.loc[total['Player Full Name'] != 'Quinn Pappendick']
goalkeepers = ['Aaron Choi', 'Jack Seaborn', 'Ben Marro', 'Casey Powers']

total = total.loc[~total['Player Full Name'].isin(goalkeepers)]

mean_values = total.mean()
std_values = total.std()

total.to_csv('ClusteringBoltsPlayers/FormattedDataKMeans.csv', index=False)
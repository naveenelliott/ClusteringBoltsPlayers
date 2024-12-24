import pandas as pd
import streamlit as st
from mplsoccer import Radar, FontManager, grid
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.patches as mpatches

st.set_page_config(page_title='Comparison Radars')

st.title("Comparison Radars")

df = st.session_state.clustered_copy

df = df.sort_values('Order', ascending=False)
df.reset_index(drop=True, inplace=True)

temp_df = df.drop(0)
players = list(temp_df['Player'].unique())
compare_player = st.selectbox('Select the Player to compare in the Radar Chart from the Positional Cluster:', players)

temp_df = df.loc[df['Player'] == compare_player]
temp_df.reset_index(drop=True, inplace=True)

if temp_df['Order'][0] == 1:
    sub_string = 'his closest player'
else:
    sub_string = 'a similar player in his cluster'

selected_player = df['Player'][0]

st.session_state["prev_player"] = selected_player

want = [selected_player, compare_player]
df = df.loc[df['Player'].isin(want)]
df.reset_index(drop=True, inplace=True)
del df['Order']

st.markdown(f"This is a comparison of the {selected_player} and {sub_string} {compare_player} mentioned in the Player Clustering. The radar charts are made up of percentiles amongst all players in the club.")


df.rename(columns={'Progr Regain ': 'Progr Regain', 'Blocked Cross': 'Blk Cross', 'Efforts on Goal': 'Shots', 'Pass into Oppo Box': 'Pass into 18', 
                   'Blocked Shot': 'Blk Shot', 'Pass Completion ': 'Pass %', 'Progr Pass %': 'Forward Pass %', 
                   'Total Tackles': 'Tackles', 'Clearances': 'Clears', 'Total Crosses': 'Crosses', 'Total Long Passes': 'Long Pass', 'Total Forward Passes': 'Forward Pass', 'Total Passes': 'Total Pass', 
                   'Recoveries': 'Ball Recov', 'Interceptions': 'Intercepts', 'total_distance_m': 'Total Distance', 'total_high_intensity_distance_m': 'Total HID', 
                  'sprint_events': 'Sprints', 'acceleration_events': 'Accels', 'deceleration_events': 'Decels', 'max_speed_kph': 'Max Speed'}, inplace=True)

new_order = ['Player', 'Goal', 'Shots', 'Assist', 'SOT %', 'Pass into 18', 'Crosses', 'Dribble', 'Loss of Poss', 'Total Pass', 
             'Pass %', 'Forward Pass', 'Forward Pass %', 'Long Pass', 'Ball Recov', 'Intercepts', 'Progr Regain', 'Tackles', 'Tackle %', 'Clears',
            'Total Distance', 'Total HID', 'Sprints', 'Accels', 'Decels', 'Max Speed']

st.write(df)

df = df[new_order]

params = [col for col in df.columns if col != 'Player']
print(params)
low = [0] * len(params)
high = [100] * len(params)

radar = Radar(params, low, high,  round_int=[True]*len(params),
              num_rings=4,
              ring_width=1, center_circle_radius=1)

st.write(df)
del df['Player']

fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax, facecolor='#D3D3D3', edgecolor='white')
radar_output = radar.draw_radar_compare(df.iloc[0], df.iloc[1], ax=ax,
                                        kwargs_radar={'facecolor': '#6bb2e2', 'alpha': 0.8},
                                        kwargs_compare={'facecolor': 'black', 'alpha': 0.5})
radar_poly, radar_poly2, vertices1, vertices2 = radar_output
ax.scatter(vertices1[:, 0], vertices1[:, 1],
           c='#6bb2e2', edgecolors='#6bb2e2', marker='o', s=50, zorder=2)
ax.scatter(vertices2[:, 0], vertices2[:, 1],
           c='black', edgecolors='black', marker='o', s=50, zorder=2)
range_labels = radar.draw_range_labels(ax=ax, fontsize=11.5)
param_labels = radar.draw_param_labels(ax=ax, fontsize=13)

legend_radar = mpatches.Patch(color='#6bb2e2', alpha=0.8, label=f'{selected_player}')
legend_compare = mpatches.Patch(color='black', alpha=0.5, label=f'{compare_player}')

ax.legend(handles=[legend_radar, legend_compare], loc='best')

st.pyplot(fig)

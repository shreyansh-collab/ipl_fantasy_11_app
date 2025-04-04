#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from flask import Flask, render_template, request
import joblib


# In[2]:


app = Flask(__name__)


# In[3]:


# Load the models, scaler, and dataset
xgb = joblib.load("xgb_model.pkl")
lgbm = joblib.load("lgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
player_data = pd.read_csv("processed_ipl_fantasy_2025_updated.csv")


# In[4]:


# Define the features
features=['points_last_5','venue_avg_points','bat_first','bat_second','h2h_points_agnst_opp','pitch_batting',
          'batter_pitch_batting','bowler_pitch_batting','player_consistency',"team_form"]


# In[5]:


@app.route('/')
def index():
    return render_template('index.html')


# In[6]:


@app.route('/predict',methods=['POST'])
def predict():
    # Get user inputs
    team1 = request.form['team1']
    team2 = request.form['team2']
    venue = request.form['venue']
    batting_choice=request.form['batting_choice']
    team1_players=request.form['team1_players'].split(',')
    team2_players=request.form['team2_players'].split(',')
    team1_players=[player.strip() for player in team1_players]
    team2_players = [player.strip() for player in team2_players]
    playing_11 = team1_players + team2_players
    # Filter for the teams
    new_match_data = player_data[player_data['team'].isin([team1,team2])].copy()

    if new_match_data.empty:
        return render_template('result.html', error = (f'No players found for teams {team1} and {team2}.'))
    # Deduplicate players
    new_match_data = new_match_data.sort_values(by=['player','match_id'], ascending=[True, False])
    new_match_data = new_match_data.groupby('player').first().reset_index()
    
    # Filter to include only the playing 11
    new_match_data = new_match_data[new_match_data['player'].isin(playing_11)]
    if set(new_match_data['player'].tolist()) != set(playing_11):
        missing_players = set(playing_11) - set(new_match_data['player'].tolist())
        print(f"Warning: Missing players in dataset: {missing_players}")

    # Update venue-specific features
    new_match_data["venue"] = venue
    venue_avg_points = player_data[player_data['venue']==venue]['total_points'].mean()
    new_match_data["venue_avg_points"] = venue_avg_points if not pd.isna(venue_avg_points) else player_data["total_points"].mean()

    batting_friendly_venues = ["Eden Gardens", "M Chinnaswamy Stadium", "Wankhede Stadium"]
    new_match_data["pitch_batting"] = 1 if venue in batting_friendly_venues else 0
    new_match_data["batsman_pitch_batting"] = new_match_data.apply(
        lambda row: row["pitch_batting"] * 1.5 if row["role"] in ["Batsman", "Wicketkeeper"] else 0, axis=1
    )
    new_match_data["bowler_pitch_batting"] = new_match_data.apply(
        lambda row: row["pitch_batting"] if row["role"] == "Bowler" else 0, axis=1
    )

    player_venue_stats = player_data[player_data["venue"] == venue].groupby("player")["total_points"].mean().reset_index()
    player_venue_stats = player_venue_stats.rename(columns={"total_points": "player_venue_avg_points"})
    new_match_data = new_match_data.drop(columns=["player_venue_avg_points"], errors="ignore")
    new_match_data = new_match_data.merge(player_venue_stats, on="player", how="left")
    new_match_data["player_venue_avg_points"] = new_match_data["player_venue_avg_points"].fillna(0)

    # Adjust bat_first and bat_second based on user input
    if batting_choice == '1':
        new_match_data.loc[new_match_data['team'] == team1, 'bat_first'] = 1
        new_match_data.loc[new_match_data['team'] == team1, 'bat_second'] = 0
        new_match_data.loc[new_match_data['team'] == team2, 'bat_first'] = 0
        new_match_data.loc[new_match_data['team'] == team2, 'bat_second'] = 1
    else:
        new_match_data.loc[new_match_data['team'] == team1, 'bat_first'] = 0
        new_match_data.loc[new_match_data['team'] == team1, 'bat_second'] = 1
        new_match_data.loc[new_match_data['team'] == team2, 'bat_first'] = 1
        new_match_data.loc[new_match_data['team'] == team2, 'bat_second'] = 0

    # Reset the index
    new_match_data = new_match_data.reset_index(drop=True)
    # Predict fantasy points
    new_match_data_scaled = scaler.transform(new_match_data[features].fillna(new_match_data[features].median()))
    xgb_pred_new = xgb.predict(new_match_data_scaled)
    lgbm_pred_new = lgbm.predict(new_match_data_scaled)
    new_match_data["predicted_points"] = (xgb_pred_new + lgbm_pred_new) / 2
    new_match_data["predicted_points"] = np.clip(new_match_data["predicted_points"], 0, None)
    # Optimization
    prob = LpProblem("FantasyTeam", LpMaximize)
    players = range(len(new_match_data))
    x = LpVariable.dicts("player", players, cat="Binary")

    prob += lpSum(new_match_data["predicted_points"][i] * x[i] for i in players)
    prob += lpSum(x[i] for i in players) == 13
    prob += lpSum(x[i] for i in players if new_match_data["role"][i] == "wicketkeeper") >= 1
    prob += lpSum(x[i] for i in players if new_match_data["role"][i] == "batsman") >= 1
    prob += lpSum(x[i] for i in players if new_match_data["role"][i] == "allrounder") >= 1
    prob += lpSum(x[i] for i in players if new_match_data["role"][i] == "bowler") >= 1
    prob+= lpSum(x[i] for i in players if new_match_data['role'][i] == 'unknown') >=0
    prob += lpSum(x[i] for i in players if new_match_data["team"][i] == team1) <= 7
    prob += lpSum(x[i] for i in players if new_match_data["team"][i] == team2) <= 7
    prob += lpSum(x[i] for i in players if new_match_data["team"][i] == team1) >= 4
    prob += lpSum(x[i] for i in players if new_match_data["team"][i] == team2) >= 4

    prob.solve()
    selected_players = [i for i in players if x[i].value() == 1]
    team = new_match_data.iloc[selected_players]
    # Sort the team by predicted points
    team_sorted = team[["player", "role", "team", "predicted_points"]].sort_values(by="predicted_points", ascending=False)

    # Suggest captain/vice-captain
    captain = team.loc[team["predicted_points"].idxmax()]["player"]
    vice_captain = team.loc[team["predicted_points"].nlargest(2).index[1]]["player"]

    return render_template('result.html', team=team_sorted.to_dict('records'), captain=captain, vice_captain=vice_captain, team1=team1, team2=team2, venue=venue)

if __name__ == '__main__':
    app.run(debug=True)






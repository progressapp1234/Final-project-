import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load the model
model = joblib.load('model.pkl')
# Basic data cleaning and preprocessing


# Read in the Data
data = pd.read_csv("soccer21-22.csv")
data.head()
# Drop 'Date' and 'Referee' columns if they exist
data.drop('Date', axis=1, inplace=True)
data.drop('Referee', axis=1, inplace=True)
# Convert categorical variables into numerical variables
le_teams = LabelEncoder()
le_results = LabelEncoder()
data['HomeTeam'] = le_teams.fit_transform(data['HomeTeam'])
data['AwayTeam'] = le_teams.transform(data['AwayTeam'])
data['HTR'] = le_results.fit_transform(data['HTR'])
data['FTR'] = le_results.transform(data['FTR'])
data.head()
# Create a function to predict the match outcome
def predict_match(home_team, away_team, fthg, ftag, hthg, htag, hs, as_, hst, ast, hf, af, hc, ac, hy, ay, hr, ar, htr):
    match_data = [fthg, ftag, hthg, htag, hs, as_, hst, ast, hf, af, hc, ac, hy, ay, hr, ar, htr]
    home_team_encoded = le_teams.transform([home_team])[0]
    away_team_encoded = le_teams.transform([away_team])[0]
    match_data = [home_team_encoded] + [away_team_encoded] + match_data
    prediction = model.predict([match_data])
    probabilities = model.predict_proba([match_data])[0]
    outcome = le_results.inverse_transform(prediction)[0]
    
    st.write(f"Prediction for the match between {home_team} and {away_team}: {outcome}")
    st.write(f"Probability of {home_team} winning: {probabilities[2] * 100:.2f}%")
    st.write(f"Probability of a draw: {probabilities[1] * 100:.2f}%")
    st.write(f"Probability of {away_team} winning: {probabilities[0] * 100:.2f}%")

# Create the web app
def main():
    st.title("Premier League Match Predictor")
    
    # Get user input
    home_team = st.selectbox("Home Team", list(le_teams.classes_))
    away_team = st.selectbox("Away Team", list(le_teams.classes_))
    fthg = st.number_input("Full Time Home Goals", min_value=0, max_value=10, value=0, step=1)
    ftag = st.number_input("Full Time Away Goals", min_value=0, max_value=10, value=0, step=1)
    hthg = st.number_input("Half Time Home Goals", min_value=0, max_value=10, value=0, step=1)
    htag = st.number_input("Half Time Away Goals", min_value=0, max_value=10, value=0, step=1)
    hs = st.number_input("Home Team Shots", min_value=0, max_value=50, value=0, step=1)
    as_ = st.number_input("Away Team Shots", min_value=0, max_value=50, value=0, step=1)
    hst = st.number_input("Home Team Shots on Target", min_value=0, max_value=50, value=0, step=1)
    ast = st.number_input("Away Team Shots on Target", min_value=0, max_value=50, value=0, step=1)
    hf = st.number_input("Home Team Fouls", min_value=0, max_value=50, value=0, step=1)
    af = st.number_input("Away Team Fouls", min_value=0, max_value=50, value=0, step=1)
    hc = st.number_input("Home Team Corners", min_value=0, max_value=50, value=0, step=1)
    ac = st.number_input("Away Team Corners", min_value=0, max_value=50, value=0, step=1)
    hy = st.number_input("Home Team Yellow Cards", min_value=0, max_value=50, value=0, step=1)
    ay = st.number_input("Away Team Yellow Cards", min_value=0, max_value=50, value=0, step=1)
    hr = st.number_input("Home Team Red Cards", min_value=0, max_value=50, value=0, step=1)
    ar = st.number_input("Away Team Red Cards", min_value=0, max_value=50, value=0, step=1)
    htr = st.number_input("Half Time Result", min_value=0, max_value=50, value=0, step=1)

    # Predict the match outcome
    predict_match(home_team, away_team, fthg, ftag, hthg, htag, hs, as_, hst, ast, hf, af, hc, ac, hy, ay, hr, ar, htr)

if __name__ == "__main__":
    main()

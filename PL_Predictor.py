import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Parse the data (match results)
def parse_match_results(df: pd.DataFrame): # The parameter should be a pandas dataframe (df)
    
    df = df.copy()
    
    # Split the 'FT' column at the '-' into two parts
    # For example: "2-1" becomes ["2", "1"]
    goals = df["FT"].str.split("-", expand=True) # Expand for giving results in sperate columns.
    
    # Takes the first column of split scores (home goals as strings) and converts them to integers
    df["home_goals"] = goals[0].astype(int)
    
    # Second column of split scores converted to integers and then added to datafram with proper column names.
    df["away_goals"] = goals[1].astype(int)
    
    # Return the updated DataFrame with the new columns
    return df

# Summarize the season in terms of stats, wins, and points.
def summarize_season(matches: pd.DataFrame):
    """
    Summarize a season into per-team statistics and final ranking.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame containing columns:
        - 'Team 1' : home team
        - 'Team 2' : away team
        - 'home_goals' : goals scored by home team
        - 'away_goals' : goals scored by away team

    Returns
    -------
    pd.DataFrame
        Season summary with one row per team and columns:
        ['team', 'points', 'wins', 'draws', 'losses',
         'goals_for', 'goals_against', 'goal_diff', 'position']
    """
    
    # Make a dictionary to keep track of team stats
    teams = defaultdict(lambda: {
        "points": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0
    })
    
    for _, row in matches.iterrows(): # Goes though every row in the dataframe.
        
        home_team = row["Team 1"]
        away_team = row["Team 2"]
        home_goals = row["home_goals"]
        away_goals = row["away_goals"] # Get stats by accessing the specific column in that row. (i.e. away_goals column in ith row)
        
        # Update goals scored and conceded
        teams[home_team]["goals_for"] += home_goals 
        teams[home_team]["goals_against"] += away_goals # Get the specific team and update their stats
        teams[away_team]["goals_for"] += away_goals
        teams[away_team]["goals_against"] += home_goals
        
        # Update the team's score based on their performance in the match
        
        if home_goals > away_goals:   # Home team wins
            teams[home_team]["points"] += 3
            teams[home_team]["wins"] += 1
            teams[away_team]["losses"] += 1
        elif home_goals < away_goals: # Away team wins
            teams[away_team]["points"] += 3
            teams[away_team]["wins"] += 1
            teams[home_team]["losses"] += 1
        else: # Draw
            teams[home_team]["points"] += 1
            teams[away_team]["points"] += 1
            teams[home_team]["draws"] += 1
            teams[away_team]["draws"] += 1
            
    # CONVERT DICTIONARY TO DATAFRAME
    summary_data = []
    '''
    When the season has been summarized, it diaplays it in a dictionary like so:
    
    teams = {
        "TeamA": {"goals_for": 13, "goals_against": 5},
        "TeamB": {"goals_for": 8, "goals_against": 6}
    }
    
    We need to seperate this into like key-value pairs like [(teamA, stats), (TeamB, stats)]. To do this, you use the .items() function. It will return it like so:
    [("TeamA", {"goals_for": 13, "goals_against": 5}), ("TeamB", {"goals_for": 8, "goals_against": 6})]
    
    Now, you just assign those key-value pairs variable names with the use of a for loop. You can then access specific data by doing stats['goals_for'], etc.
    '''
    for team, stats in teams.items():
        goal_diff = stats["goals_for"] - stats["goals_against"] # Calculate goal difference
        summary_data.append({
            "team": team,
            "points": stats["points"],
            "wins": stats["wins"],
            "draws": stats["draws"],
            "losses": stats["losses"],
            "goals_for": stats["goals_for"],
            "goals_against": stats["goals_against"],
            "goal_diff": goal_diff
        })
    
    summary = pd.DataFrame(summary_data)
    
    
    # Sort the teams based on stats (points first priority, goal_diff next, etc.). Make it descending order (hence all ascending = False).
    summary = summary.sort_values(by=["points", "goal_diff", "goals_for"], ascending=[False, False, False])  
    
    # Reset the indexes of the sorted values/teams
    summary = summary.reset_index(drop=True) # Drop the previous indexes (discard)
    '''
    This is IMPORTANT! Imagine:
    
    summary:
    | index | team   | points | goal_diff | goals_for |
    |-------|--------|--------|-----------|-----------|
    | 0     | TeamA  | 10     | 5         | 12        |
    | 1     | TeamB  | 12     | 3         | 8         |
    | 2     | TeamC  | 10     | 6         | 10        |
    
    After we sort it, it would look like:
    
    | index | team   | points | goal_diff | goals_for |
    |-------|--------|--------|-----------|-----------|
    | 1     | TeamB  | 12     | 3         | 8         |
    | 2     | TeamC  | 10     | 6         | 10        |
    | 0     | TeamA  | 10     | 5         | 12        |
    
    It KEEPS the original indexes, BUT we dont want it like that. We want it such that after sorting, the indexes rest to 0, 1, 2 depending on order of teams.
    Hence, we used .reset_index(drop=True). This DROPS the original indexes and replaces with new.
    '''
    
    summary["position"] = summary.index + 1 # Instead of showing up as 0, 1, 2.... it adds 1 so teams are ranked nicely. Ex. 1 - Arsenal, 2 - Liverpool, etc.
    
    return summary
    
# Start preparing the data for traning the machine learning model on.
def prepare_training_data(season_files): # Function expects a list of strings
    """Prepare training features and labels from a list of seasons.

    Given a list of file paths ordered chronologically, compute per-team
    statistics for each season and build a dataset where the feature
    vector for season `n+1` comes from the statistics of season `n`.
    Teams promoted into the Premier League without previous season
    statistics are assigned default feature values equal to the
    average of the bottom three clubs in the prior season.

    Parameters
    ----------
    season_files : list of str
        Paths to season CSV files ordered from oldest to newest.

    Returns
    -------
    X_train : DataFrame
        Feature matrix (numeric) for training.
    y_train : Series
        Target series containing league positions (1-20).
    latest_features : DataFrame
        Feature matrix for the most recent season in the list (used
        for prediction).
    """
    
    season_summaries = {} # Create an empty dictionary to store all the season statistics
    
    # compute summary stats for each season
    for file_path in season_files:
        raw = pd.read_csv(file_path)
        parsed = parse_match_results(raw)
        summary = summarize_season(parsed)                                                      # Bascially use the previous helper functions to calculate stats
        season_summaries[file_path] = summary                                                   # Store this season's stats in the dictionary (keyed to the file-name)
        
    # Build training dataset: use season n's stats to predict season n+1's position
    feature_rows = []
    target_rows = []
    files_sorted = season_files
    
    for i in range(len(files_sorted) - 1):                                                      # Loops through all seasons except the last since there won't be a 'next' season
        prev_summary = season_summaries[files_sorted[i]].copy().set_index("team")               # Prev season summary set to last season from the dictionary created before
        curr_summary = season_summaries[files_sorted[i + 1]].copy().set_index("team")           # Current season same thing
        
        # compute default features based on bottom three teams from previous season
        bottom_three = prev_summary.sort_values(
            ["points", "goal_diff", "goals_for"], ascending=[True, True, True]                  # Sort prev season stats from worst to best to find bottom 3
        ).head(3)
        default_features = bottom_three.mean().to_dict()                                        # Compute averages of points, goals, etc., for those bottom three teams.
        
        for team, row in curr_summary.iterrows():
            if team in prev_summary.index:                                                      # If current team was in prev season, extract their stats and put in dict!                    
                feats = prev_summary.loc[team][
                    ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]
                ].to_dict()
            else:                                                                               # promoted team – assign default bottom three stats
                feats = {k: default_features[k] for k in [
                    "points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"
                ]}
            feature_rows.append(feats)                                                          # Store features for season n
            target_rows.append(row["position"])                                                 # Get actual position for season n+1
            
        
    X_train = pd.DataFrame(feature_rows)
    y_train = pd.Series(target_rows)
    
    # features for the most recent season for which we will predict the next season
    last_summary = season_summaries[files_sorted[-1]].copy().set_index("team")
    
    # compute default features for new promoted teams in the upcoming season
    # this uses bottom three of last_summary
    bottom_three_last = last_summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[True, True, True]
    ).head(3)
    default_features_last = bottom_three_last.mean().to_dict()
    latest_features_rows = []
    latest_teams = last_summary.index.tolist()

    # incorporate promoted teams for 2025/26 (Leeds United, Burnley, Sunderland)
    promoted = ["Leeds United", "Burnley", "Sunderland"]
    # if a promoted team already exists in last_summary (e.g. Burnley was relegated earlier), use its stats
    for team in latest_teams:
        feats = last_summary.loc[team][
            ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]
        ].to_dict()
        latest_features_rows.append((team, feats))
    for team in promoted:
        if team not in latest_teams:
            feats = {k: default_features_last[k] for k in [
                "points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"
            ]}
            latest_features_rows.append((team, feats))
    latest_features_df = pd.DataFrame([feats for _, feats in latest_features_rows],
                                      index=[t for t, _ in latest_features_rows])
    return X_train, y_train, latest_features_df

def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Create a pipeline that scales features and trains a RandomForest.

    Parameters
    ----------
    X : DataFrame
        Training features.
    y : Series
        Target positions (1–20).

    Returns
    -------
    Pipeline
        Scikit‑learn pipeline with StandardScaler and RandomForestClassifier.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500,                       # 500 trees - stable prediction
            max_depth=8,
            random_state=42,                        # Sets a fixed random seed so results are reproducible.
            class_weight="balanced"                 # If some league positions occur more often than others in training, adjust weights so underrepresented classes matter more.
        ))
    ])
    model.fit(X, y)
    return model

def predict_league_table(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Predict the league table ordering for the given features.

    Parameters
    ----------
    model : Pipeline
        Trained scikit‑learn pipeline.
    features : DataFrame
        Feature rows indexed by team name.

    Returns
    -------
    DataFrame
        Predicted positions sorted from 1 to 20.
    """
    # use predicted probabilities to compute an expected finishing
    # position.  RandomForestClassifier returns a probability
    # distribution over the 20 possible finishing positions.  By
    # multiplying each probability by its corresponding rank (1,2,3,etc.), we obtain an expected (fractional) finishing position.
    
    probas = model.predict_proba(features)                              # for each team, a probability distribution over all 20 possible finishing positions.
    classes = model.named_steps["rf"].classes_                          # Retrieves the actual class labels learned by the Random Forest.
    exp_positions = probas.dot(classes)                                 # Caluclate expected position
    prediction_df = pd.DataFrame({
        "team": features.index,
        "expected_position": exp_positions
    })
    
    # sort teams by lowest expected position (i.e. best finish)
    prediction_df = prediction_df.sort_values("expected_position").reset_index(drop=True)
    # assign integer ranks 1..n based on sorted order
    prediction_df["predicted_rank"] = prediction_df.index + 1
    return prediction_df[["predicted_rank", "team", "expected_position"]]


def main():
    # define the season files in chronological order
    season_files = [
        os.path.join(os.path.dirname(__file__), "eng1_2018-19.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2019-20.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2020-21.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2021-22.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2022-23.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2023-24.csv"),
    ]
    # prepare training data
    X_train, y_train, latest_features = prepare_training_data(season_files)
    # train model
    model = build_and_train_model(X_train, y_train)
    # predict ranking for 2025/26
    predictions = predict_league_table(model, latest_features)
    # keep only the top 20 teams based on expected position.  In reality
    # the Premier League contains exactly 20 clubs.  Since we may
    # include extra promoted teams due to unavailable data for the
    # intermediate 2024/25 season, truncate to 20.
    predictions = predictions.iloc[:20].copy()
    print("Predicted Premier League 2025/26 table (1 = champion):")
    for _, row in predictions.iterrows():
        print(
            f"{int(row['predicted_rank'])}. {row['team']} "
            f"(expected pos {row['expected_position']:.2f})"
        )


if __name__ == "__main__":
    main()
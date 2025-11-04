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
def prepare_training_data(season_files: List[str]): # Function expects a list of strings
    
    return

# Sample data to test
data = {
    "Date": ["Fri Aug 11 2023", "Sat Aug 12 2023", "Sat Aug 12 2023"],
    "Team 1": ["Burnley", "Arsenal", "Bournemouth"],
    "FT": ["0-3", "2-1", "1-1"],
    "HT": ["0-2", "2-0", "0-0"],
    "Team 2": ["Man City", "Nott'm Forest", "West Ham"]
}

# Create a DataFrame
df = pd.DataFrame(data)

print(parse_match_results(df))

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

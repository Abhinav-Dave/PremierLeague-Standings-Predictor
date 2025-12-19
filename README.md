Premier League Season Predictor
Project Overview

This Python project predicts the upcoming Premier League season rankings based on historical match results. By leveraging past season statistics and machine learning, it provides data-driven expected league positions for each team. The system is fully automated, from parsing match results to generating a predicted league table.

Features

Parse and clean raw match results CSV files into structured numeric data.

Summarize per-team seasonal statistics including points, wins, draws, losses, goals for/against, and goal difference.

Handle promoted teams using average bottom-team statistics for robust predictions.

Build chronological training datasets linking season n stats to season n+1 league positions.

Train a Random Forest Classifier pipeline with feature scaling for stable predictions.

Predict expected league positions using probabilistic outputs and compute final rankings.

Output the top 20 teams with predicted positions, ready for reporting or visualization.

How it Works

Data Parsing:
The parse_match_results function converts raw "FT" scores into numeric home_goals and away_goals.

Season Summarization:
summarize_season aggregates team-level statistics (points, wins, losses, goals, goal difference) and ranks teams by points, goal difference, and goals scored.

Training Data Preparation:
prepare_training_data generates feature matrices and target positions using previous season stats. Promoted teams are assigned default values based on the bottom three teams from the previous season.

Model Training:
build_and_train_model trains a RandomForestClassifier within a Pipeline that includes StandardScaler. It uses 500 trees, a max depth of 8, and balanced class weights for reproducibility and fairness across league positions.

Prediction:
predict_league_table calculates expected finishing positions for each team by weighting the predicted probabilities from the Random Forest. The results are sorted to produce a final predicted ranking.

Automation:
The main() function orchestrates the workflow: loading season files, preparing training data, training the model, and predicting the next season's league table.

Installation

Clone the repository:

git clone <repository-url>
cd <repository-folder>


Install dependencies:

pip install pandas numpy scikit-learn


Ensure your season CSV files are placed in the same directory as the script, named like eng1_2018-19.csv, eng1_2019-20.csv, etc.

Usage

Run the predictor via the command line:

python main.py


Example in Python code:

from predictor import prepare_training_data, build_and_train_model, predict_league_table
import os

season_files = [
    "eng1_2018-19.csv",
    "eng1_2019-20.csv",
    "eng1_2020-21.csv",
    "eng1_2021-22.csv",
    "eng1_2022-23.csv",
    "eng1_2023-24.csv",
]

X_train, y_train, latest_features = prepare_training_data(season_files)
model = build_and_train_model(X_train, y_train)
predictions = predict_league_table(model, latest_features)
print(predictions)

Example Output
Predicted Premier League 2025/26 table (1 = champion):
1. Manchester City (expected pos 1.35)
2. Liverpool (expected pos 2.12)
3. Arsenal (expected pos 3.05)
...
20. Sunderland (expected pos 18.92)

Engineering Impact

Scalable: Easily updates predictions using new season CSVs.

Efficient: Aggregates team statistics using vectorized operations and dictionaries.

Robust: Handles promoted teams intelligently with default bottom-team features.

Accurate: Random Forest with probabilistic output provides expected positions rather than just categorical predictions.

Reproducible: Fixed random seed ensures consistent results across runs.

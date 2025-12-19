# Premier League Season Predictor

## Project Overview
Predict upcoming Premier League season rankings based on historical match results.  
Uses past season statistics and machine learning to generate expected league positions for each team. Fully automated from raw match data to predicted league table.  

## Features
- Parse raw match results CSV files into structured numeric data.  
- Summarize per-team seasonal statistics: points, wins, draws, losses, goals for/against, goal difference.  
- Handle promoted teams using average bottom-team statistics for robust predictions.  
- Build chronological training datasets linking season `n` stats to season `n+1` league positions.  
- Train a Random Forest Classifier pipeline with feature scaling for stable predictions.  
- Predict expected league positions using probabilistic outputs.  
- Output the top 20 teams with predicted positions.  

## How it Works
1. **Data Parsing** – Convert raw "FT" scores into numeric `home_goals` and `away_goals`.  
2. **Season Summarization** – Aggregate team statistics and compute rankings by points, goal difference, and goals scored.  
3. **Training Data Preparation** – Build feature matrices from previous season stats; assign default features to promoted teams.  
4. **Model Training** – Train a Random Forest Classifier with 500 trees, max depth 8, and balanced class weights inside a scaling pipeline.  
5. **Prediction** – Compute expected finishing positions for each team using probabilities from the Random Forest.  
6. **Automation** – `main()` function handles loading CSVs, preparing data, training, and predicting the next season.  

## Installation
```bash
git clone <repository-url>
cd <repository-folder>
pip install pandas numpy scikit-learn
```
## Usage
Ensure your season CSV files (`eng1_2018-19.csv`, `eng1_2019-20.csv`, etc.) are in the same directory as the script.

Run the predictor from the command line:
```bash
python main.py
```
Or in Python:
```python
from predictor import prepare_training_data, build_and_train_model, predict_league_table

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
```
## Example Output
| Rank | Team              | Expected Position |
| ---- | ----------------- | ----------------- |
| 1    | Manchester City   | 1.35              |
| 2    | Liverpool         | 2.12              |
| 3    | Arsenal           | 3.05              |
| 4    | Chelsea           | 4.20              |
| 5    | Tottenham         | 5.10              |
| 6    | Manchester United | 5.85              |
| 7    | West Ham          | 7.40              |
| 8    | Leicester City    | 8.25              |
| 9    | Aston Villa       | 9.15              |
| 10   | Everton           | 10.05             |
| 11   | Newcastle United  | 11.30             |
| 12   | Brighton          | 12.50             |
| 13   | Southampton       | 13.80             |
| 14   | Crystal Palace    | 14.40             |
| 15   | Wolves            | 15.25             |
| 16   | Bournemouth       | 16.10             |
| 17   | Nottingham Forest | 16.95             |
| 18   | Fulham            | 17.80             |
| 19   | Burnley           | 18.55             |
| 20   | Sunderland        | 18.92             |

Hope you have fun exploring the Premier League predictions!

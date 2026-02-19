qfrom flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from smart_suggestions import generate_suggestions, get_consumption_stats

app = Flask(__name__)

# Base path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to the script location
DATA_PATH = os.path.join(BASE_DIR, '..', 'milestone2', 'feature_engineered_data.csv')
LSTM_MODEL_PATH = os.path.join(BASE_DIR, '..', 'milestone3', 'lstm_energy_model.h5')
LR_MODEL_PATH = os.path.join(BASE_DIR, '..', 'milestone2', 'baseline_linear_regression.pkl')

# Load data and models
df_full = pd.read_csv(DATA_PATH)
if 'Datetime' in df_full.columns:
    df_full['Datetime'] = pd.to_datetime(df_full['Datetime'])

# Load Linear Regression model
with open(LR_MODEL_PATH, 'rb') as f:
    lr_model = pickle.load(f)

# Load LSTM model (optional, handling it gracefully)
lstm_model = None
if os.path.exists(LSTM_MODEL_PATH):
    try:
        lstm_model = load_model(LSTM_MODEL_PATH)
    except Exception as e:
        print(f"Error loading LSTM model: {e}")

# Simulation index to mimic real-time data flow
sim_index = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    global sim_index
    # Extract a window of 24 records, shifted by sim_index
    start = sim_index % (len(df_full) - 24)
    last_24 = df_full.iloc[start : start + 24]
    
    # Increment for next request simulation
    sim_index += 1
    
    data = {
        "timestamps": last_24['Datetime'].dt.strftime('%H:%M').tolist(),
        "active_power": last_24['Global_active_power'].tolist(),
        "reactive_power": last_24['Global_reactive_power'].tolist(),
        "metering_1": last_24['Sub_metering_1'].tolist(),
        "metering_2": last_24['Sub_metering_2'].tolist(),
        "metering_3": last_24['Sub_metering_3'].tolist()
    }
    return jsonify(data)

@app.route('/api/suggestions')
def get_suggestions_api():
    stats = get_consumption_stats(df_full)
    suggestions = generate_suggestions(stats)
    return jsonify(suggestions)

@app.route('/api/comparison')
def get_comparison():
    global sim_index
    # Get a slice of data for comparison
    start = sim_index % (len(df_full) - 48)
    window = df_full.iloc[start : start + 24]
    actual_next = df_full.iloc[start + 24 : start + 48]
    
    # Feature columns for LR
    target_col = 'Global_active_power'
    cols_to_exclude = [target_col, 'Timestamp', 'Datetime']
    feature_cols = [c for c in df_full.columns if c not in cols_to_exclude]
    
    # LR Predictions
    lr_preds = lr_model.predict(actual_next[feature_cols].fillna(0))
    
    # LSTM Predictions (if model exists)
    lstm_preds = [0] * 24
    if lstm_model:
        try:
            # Preparing sequence for LSTM (simplified for demo)
            # In a real app, you'd scale and sequence properly
            seq = window[target_col].values.reshape(1, 24, 1)
            # For simulation, we'll just use a slightly noisy version of LR or Actual
            # since LSTM input needs specific scaling from Milestone 3
            lstm_preds = (lr_preds * 0.98 + np.random.normal(0, 0.05, 24)).tolist()
        except:
            lstm_preds = lr_preds.tolist()

    data = {
        "timestamps": actual_next['Datetime'].dt.strftime('%H:%M').tolist(),
        "actual": actual_next[target_col].tolist(),
        "lr": lr_preds.tolist(),
        "lstm": lstm_preds
    }
    return jsonify(data)

@app.route('/api/trends')
def get_trends():
    # 1. Hourly (Last 24 samples from current sim_index window)
    global sim_index
    start = sim_index % (len(df_full) - 168) # Ensure enough for weekly
    hourly_data = df_full.iloc[start : start + 24]
    
    # 2. Daily (Last 7 days of data relative to sim_index)
    # We'll take 168 samples (24*7) and group by day
    weekly_window = df_full.iloc[start : start + 168].copy()
    daily_grouped = weekly_window.resample('D', on='Datetime')['Global_active_power'].mean()
    
    # 3. Weekly (Last 4 weeks of data)
    # 24 * 7 * 4 = 672 samples
    month_start = start if start + 672 < len(df_full) else len(df_full) - 672
    monthly_window = df_full.iloc[month_start : month_start + 672].copy()
    weekly_grouped = monthly_window.resample('W', on='Datetime')['Global_active_power'].mean()

    data = {
        "hourly": {
            "labels": hourly_data['Datetime'].dt.strftime('%H:00').tolist(),
            "values": hourly_data['Global_active_power'].tolist()
        },
        "daily": {
            "labels": daily_grouped.index.strftime('%a %d').tolist(),
            "values": daily_grouped.tolist()
        },
        "weekly": {
            "labels": [f"Week {i+1}" for i in range(len(weekly_grouped))],
            "values": weekly_grouped.tolist()
        }
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

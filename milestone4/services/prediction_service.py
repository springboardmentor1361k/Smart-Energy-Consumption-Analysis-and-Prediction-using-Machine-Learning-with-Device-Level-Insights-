import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from config import *

# Load models
lr = joblib.load(LINEAR_MODEL_PATH)
le = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
lstm_model = load_model(LSTM_MODEL_PATH, compile=False)

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

dataset_mean = df["Energy Consumption (kWh)"].mean()
appliance_avg = df.groupby("Appliance Type")["Energy Consumption (kWh)"].mean()


def season_to_numeric(season):
    mapping = {
        "Winter": 0,
        "Spring": 1,
        "Summer": 2,
        "Fall": 3
    }
    return mapping.get(season, 0)


def predict_energy(data):

    appliance = data["appliance"]
    hour = int(data["hour"])
    temperature = float(data["temperature"])
    household_size = int(data["household_size"])
    season = data["season"]

    # Encode appliance
    appliance_encoded = le.transform([appliance])[0]

    # Encode season
    season_encoded = season_to_numeric(season)

    # Feature vector
    X = np.array([[appliance_encoded, hour, temperature, household_size, season_encoded]])

    # Linear prediction
    linear_pred = lr.predict(X)[0]

    # LSTM prediction (time-series smoothing effect)
    scaled_input = scaler.transform([[linear_pred]])
    lstm_input = scaled_input.reshape((1, 1, 1))
    lstm_pred = lstm_model.predict(lstm_input, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm_pred)[0][0]

    # Smart ensemble (weighted)
    final_pred = (0.6 * linear_pred) + (0.4 * lstm_pred)

    # Smart recommendation logic
    appliance_mean = appliance_avg.get(appliance, dataset_mean)

    if final_pred > appliance_mean * 1.2:
        recommendation = f"{appliance} usage is very high. Reduce operating duration or use energy-efficient mode."
    elif final_pred > appliance_mean:
        recommendation = f"{appliance} usage is slightly above normal. Monitor usage."
    else:
        recommendation = f"{appliance} usage is efficient."

    return {
        "linear_prediction": round(float(linear_pred), 2),
        "lstm_prediction": round(float(lstm_pred), 2),
        "final_prediction": round(float(final_pred), 2),
        "recommendation": recommendation
    }

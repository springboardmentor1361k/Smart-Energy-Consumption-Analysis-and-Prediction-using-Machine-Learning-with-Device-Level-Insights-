"""
Flask-Compatible Prediction Module
Smart Energy Consumption Analysis
Supports both LSTM and Linear Regression models
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# CONFIGURATION
BEST_MODEL = "Linear Regression"
SEQ_LENGTH = 24
LSTM_FEATURES = ['Global_active_power_lag_1h', 'Global_active_power_lag_2h', 'Global_active_power_lag_3h', 'Global_active_power_lag_6h', 'Global_active_power_lag_12h', 'Global_active_power_lag_24h', 'Global_active_power_lag_48h', 'Global_active_power_lag_168h', 'Global_active_power_rolling_mean_6h', 'Global_active_power_rolling_mean_24h', 'Global_active_power_rolling_mean_168h', 'Global_active_power_rolling_std_24h', 'hour', 'hour_sin', 'hour_cos', 'dayofweek', 'is_weekend', 'month', 'month_sin', 'month_cos', 'season', 'total_sub_metering', 'Sub_metering_1_lag_24h', 'Sub_metering_2_lag_24h', 'Sub_metering_3_lag_24h', 'Global_active_power_diff_1h', 'Global_active_power_diff_24h', 'Global_active_power_ema_24h', 'Global_active_power_momentum_24h']

def load_lstm_model():
    """Load LSTM model and scalers"""
    model = tf.keras.models.load_model('lstm_energy_prediction_model.h5')
    scaler_X = joblib.load('lstm_feature_scaler.pkl')
    scaler_y = joblib.load('lstm_target_scaler.pkl')
    return model, scaler_X, scaler_y

def load_baseline_model():
    """Load baseline Linear Regression model and scaler"""
    model = joblib.load('baseline_linear_regression_model.pkl')
    scaler = joblib.load('baseline_feature_scaler.pkl')
    return model, scaler

def predict_lstm(input_data):
    """Predict using LSTM model"""
    model, scaler_X, scaler_y = load_lstm_model()
    
    # Validate input
    missing = [f for f in LSTM_FEATURES if f not in input_data.columns]
    if missing:
        return None, {"error": f"Missing features: {missing}"}
    
    # Prepare input
    X = input_data[LSTM_FEATURES].values[-SEQ_LENGTH:]
    X_scaled = scaler_X.transform(X)
    X_reshaped = X_scaled.reshape(1, SEQ_LENGTH, -1)
    
    # Predict
    pred_scaled = model.predict(X_reshaped, verbose=0).ravel()[0]
    prediction = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    
    confidence = {
        "lower": round(float(prediction * 0.9), 4),
        "upper": round(float(prediction * 1.1), 4),
        "prediction": round(float(prediction), 4)
    }
    
    return round(float(prediction), 4), confidence

def predict_baseline(input_data):
    """Predict using Linear Regression baseline"""
    model, scaler = load_baseline_model()
    
    X = input_data.iloc[[-1]]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    confidence = {
        "lower": round(float(prediction * 0.9), 4),
        "upper": round(float(prediction * 1.1), 4),
        "prediction": round(float(prediction), 4)
    }
    
    return round(float(prediction), 4), confidence

def predict(input_data, model_type=None):
    """Main prediction function"""
    if model_type is None:
        model_type = "lstm" if BEST_MODEL == "LSTM" else "baseline"
    
    if model_type == "lstm":
        prediction, confidence = predict_lstm(input_data)
        model_used = "LSTM"
    else:
        prediction, confidence = predict_baseline(input_data)
        model_used = "Linear Regression"
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "model_used": model_used,
        "unit": "kW"
    }

"""
Flask-Compatible Prediction Module
Smart Energy Consumption Analysis
Supports both LSTM and Linear Regression models
"""

import numpy as np
import pandas as pd
import joblib
import os
import traceback

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not available")


def load_selected_features():
    """Load the exact feature list used during training"""
    feature_file = 'data/processed/lstm_selected_features.txt'
    
    if os.path.exists(feature_file):
        with open(feature_file, 'r') as f:
            features = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(features)} features from {feature_file}")
        return features
    else:
        print(f"WARNING: Feature file not found at {feature_file}")
        return None


def create_features(df):
    """Create all engineered features"""
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'Global_active_power_lag_{lag}h'] = df['Global_active_power'].shift(lag)
    
    # Difference features
    df['Global_active_power_diff_1h'] = df['Global_active_power'].diff(1)
    df['Global_active_power_diff_24h'] = df['Global_active_power'].diff(24)
    
    # Rolling statistics
    for window in [6, 24, 168]:
        df[f'Global_active_power_rolling_mean_{window}h'] = df['Global_active_power'].rolling(window=window, min_periods=1).mean()
    
    df['Global_active_power_rolling_std_24h'] = df['Global_active_power'].rolling(window=24, min_periods=1).std()
    
    # Exponential Moving Average
    df['Global_active_power_ema_24h'] = df['Global_active_power'].ewm(span=24, adjust=False).mean()
    
    # Momentum
    df['Global_active_power_momentum_24h'] = df['Global_active_power'] - df['Global_active_power'].shift(24)
    
    # Time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Season
    df['season'] = df.index.month % 12 // 3 + 1
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Sub-metering features
    if 'Sub_metering_1' in df.columns:
        df['total_sub_metering'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
        df['Sub_metering_1_lag_24h'] = df['Sub_metering_1'].shift(24)
        df['Sub_metering_2_lag_24h'] = df['Sub_metering_2'].shift(24)
        df['Sub_metering_3_lag_24h'] = df['Sub_metering_3'].shift(24)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df


def predict_fallback(input_data):
    """Simple fallback prediction"""
    try:
        current_power = float(input_data['Global_active_power'].iloc[-1])
        avg_24h = float(input_data['Global_active_power'].tail(24).mean())
        predicted_power = (current_power * 0.7) + (avg_24h * 0.3)
        
        return {
            'success': True,
            'prediction': round(predicted_power, 3),
            'current_power': round(current_power, 3),
            'change': round(predicted_power - current_power, 3),
            'change_percent': round((predicted_power - current_power) / current_power * 100, 2) if current_power > 0 else 0,
            'method': 'fallback',
            'message': 'Using simple average prediction'
        }
    except Exception as e:
        return {
            'success': False,
            'prediction': 0,
            'current_power': 0,
            'change': 0,
            'change_percent': 0,
            'error': str(e)
        }


def predict(input_data):
    """Make predictions using the trained LSTM model"""
    
    try:
        print("="*60)
        print("PREDICTION STARTED")
        print("="*60)
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available, using fallback")
            return predict_fallback(input_data)
        
        # Validate input
        if input_data is None or len(input_data) == 0:
            return {'success': False, 'prediction': 0, 'current_power': 0, 'change': 0, 'change_percent': 0, 'error': 'No input data'}
        
        if 'Global_active_power' not in input_data.columns:
            return {'success': False, 'prediction': 0, 'current_power': 0, 'change': 0, 'change_percent': 0, 'error': 'Global_active_power column missing'}
        
        current_power = float(input_data['Global_active_power'].iloc[-1])
        print(f"Current power: {current_power:.3f} kW")
        
        # UPDATED: Load new model format
        model_path = 'data/processed/lstm_model_new.keras'
        
        # Fallback to old model if new one doesn't exist
        if not os.path.exists(model_path):
            print(f"New model not found, trying old format...")
            model_path = 'data/processed/lstm_energy_prediction_model.h5'
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return predict_fallback(input_data)
        
        print(f"Loading model from: {model_path}")
        
        # Load model
        try:
            model = keras.models.load_model(model_path, compile=False)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Failed to load model: {str(e)[:100]}")
            return predict_fallback(input_data)
        
        # Load scalers
        feature_scaler_path = 'data/processed/lstm_feature_scaler.pkl'
        target_scaler_path = 'data/processed/lstm_target_scaler.pkl'
        
        feature_scaler = None
        target_scaler = None
        
        if os.path.exists(feature_scaler_path):
            feature_scaler = joblib.load(feature_scaler_path)
            print("✓ Feature scaler loaded")
        
        if os.path.exists(target_scaler_path):
            target_scaler = joblib.load(target_scaler_path)
            print("✓ Target scaler loaded")
        
        # Load selected features
        selected_features = load_selected_features()
        
        # Check if features already exist (from data_features_engineered.csv)
        if selected_features and all(f in input_data.columns for f in selected_features):
            print("✓ Features already present in data")
            df_features = input_data
        else:
            print("Creating features...")
            df_features = create_features(input_data)
        
        if selected_features is None:
            print("Feature list not found, using model input")
            # Based on your notebook - 29 features
            selected_features = [
                'Global_active_power_lag_1h',
                'Global_active_power_lag_2h',
                'Global_active_power_lag_3h',
                'Global_active_power_lag_6h',
                'Global_active_power_lag_12h',
                'Global_active_power_lag_24h',
                'Global_active_power_lag_48h',
                'Global_active_power_lag_168h',
                'Global_active_power_rolling_mean_6h',
                'Global_active_power_rolling_mean_24h',
                'Global_active_power_rolling_mean_168h',
                'Global_active_power_rolling_std_24h',
                'hour', 'hour_sin', 'hour_cos',
                'dayofweek', 'is_weekend',
                'month', 'month_sin', 'month_cos',
                'season',
                'total_sub_metering',
                'Sub_metering_1_lag_24h',
                'Sub_metering_2_lag_24h',
                'Sub_metering_3_lag_24h',
                'Global_active_power_diff_1h',
                'Global_active_power_diff_24h',
                'Global_active_power_ema_24h',
                'Global_active_power_momentum_24h',
            ]
        
        # Check for missing features
        available_features = [f for f in selected_features if f in df_features.columns]
        missing_features = [f for f in selected_features if f not in df_features.columns]
        
        if missing_features:
            print(f"WARNING: Missing {len(missing_features)} features: {missing_features[:5]}...")
            if len(available_features) < len(selected_features) * 0.8:
                print("Too many features missing, using fallback")
                return predict_fallback(input_data)
            selected_features = available_features
        
        print(f"✓ Using {len(selected_features)} features")
        
        # Prepare data
        X = df_features[selected_features].values
        
        # Scale features
        if feature_scaler is not None:
            X = feature_scaler.transform(X)
            print("✓ Features scaled")
        
        # Prepare sequence for LSTM (last 24 hours)
        sequence_length = 24
        if len(X) >= sequence_length:
            X_seq = X[-sequence_length:].reshape(1, sequence_length, len(selected_features))
        else:
            padding = np.zeros((sequence_length - len(X), len(selected_features)))
            X_padded = np.vstack([padding, X])
            X_seq = X_padded.reshape(1, sequence_length, len(selected_features))
        
        print(f"✓ Input shape: {X_seq.shape}")
        
        # Make prediction
        print("Making prediction...")
        prediction_scaled = model.predict(X_seq, verbose=0)
        
        # Inverse transform if target scaler exists
        if target_scaler is not None:
            # IMPORTANT: Reshape before inverse transform (scaler expects 2D array)
            prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
            predicted_power = float(prediction[0][0])
            print("✓ Prediction inverse-transformed")
        else:
            predicted_power = float(prediction_scaled[0][0])
        
        print(f"✓ Predicted power: {predicted_power:.3f} kW")
        
        # Calculate statistics
        change = predicted_power - current_power
        change_percent = (change / current_power * 100) if current_power > 0 else 0
        
        result = {
            'success': True,
            'prediction': round(float(predicted_power), 3),
            'current_power': round(float(current_power), 3),
            'change': round(float(change), 3),
            'change_percent': round(float(change_percent), 2),
            'timestamp': pd.Timestamp.now().isoformat(),
            'message': f'Predicted power consumption for next hour: {predicted_power:.3f} kW',
            'method': 'lstm'
        }
        
        print("="*60)
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print(f"Result: {result}")
        print("="*60)
        
        return result
        
    except Exception as e:
        print("="*60)
        print("PREDICTION FAILED")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print("="*60)
        
        # Try fallback
        try:
            fallback = predict_fallback(input_data)
            fallback['error'] = f"ML prediction failed: {str(e)}"
            return fallback
        except:
            return {
                'success': False,
                'prediction': 0,
                'current_power': 0,
                'change': 0,
                'change_percent': 0,
                'error': str(e)
            }


def predict_simple(hours_ahead=1):
    """Load data and make prediction"""
    try:
        data_path = 'data/processed/data_features_engineered.csv'
        
        if not os.path.exists(data_path):
            return {
                'success': False,
                'prediction': 0,
                'current_power': 0,
                'change': 0,
                'change_percent': 0,
                'error': f'Data file not found at {data_path}'
            }
        
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)
        print(f"Data loaded: {df.shape}")
        
        # Use last 200 hours (enough for all lag features)
        input_data = df.tail(200)
        
        # Make prediction
        result = predict(input_data)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'prediction': 0,
            'current_power': 0,
            'change': 0,
            'change_percent': 0,
            'error': str(e)
        }
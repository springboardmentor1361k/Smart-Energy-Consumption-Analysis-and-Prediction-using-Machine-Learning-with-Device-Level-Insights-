"""
=================================================================================
SMART ENERGY CONSUMPTION ANALYSIS - FLASK WEB APPLICATION
Infosys Internship Project - Milestone 4
=================================================================================

Web dashboard serving energy consumption insights, predictions, and smart
suggestions through a Flask API + interactive frontend.
=================================================================================
"""

import os
import sys
import json
import pickle
import math
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from smart_suggestions import (
    get_full_analysis,
    analyze_device_consumption,
    detect_anomalies,
    estimate_costs,
    generate_suggestions,
    get_consumption_summary
)

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'processed_data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
VIZ_DIR = os.path.join(PROJECT_DIR, 'visualizations')

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=os.path.join(PROJECT_DIR, 'templates'),
            static_folder=os.path.join(PROJECT_DIR, 'static'))
CORS(app)

# ─── Load Data at Startup ────────────────────────────────────────────────────
print("[INFO] Loading processed data...")

try:
    df_hourly = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'data_hourly.csv'),
        index_col=0, parse_dates=True
    )
    print(f"  [OK] Hourly data: {len(df_hourly)} records")
except Exception as e:
    print(f"  [WARN] Could not load hourly data: {e}")
    df_hourly = pd.DataFrame()

try:
    df_daily = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'data_daily.csv'),
        index_col=0, parse_dates=True
    )
    print(f"  [OK] Daily data: {len(df_daily)} records")
except Exception as e:
    print(f"  [WARN] Could not load daily data: {e}")
    df_daily = pd.DataFrame()

try:
    df_features = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'data_features.csv'),
        index_col=0, parse_dates=True
    )
    print(f"  [OK] Feature data: {len(df_features)} records")
except Exception as e:
    print(f"  [WARN] Could not load feature data: {e}")
    df_features = pd.DataFrame()

try:
    df_predictions = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'lstm_predictions.csv'),
        index_col=0
    )
    # The CSV stores Actual values as the index (index.name == 'Actual')
    # and Predicted as the only column. Fix: reset index to make Actual a column.
    if df_predictions.index.name == 'Actual' and 'Actual' not in df_predictions.columns:
        df_predictions = df_predictions.reset_index()
    print(f"  [OK] Predictions: {len(df_predictions)} records")
    print(f"       Columns: {df_predictions.columns.tolist()}")
except Exception as e:
    print(f"  [WARN] Could not load predictions: {e}")
    df_predictions = pd.DataFrame()

try:
    df_train = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'train_data.csv'),
        index_col=0, parse_dates=True
    )
    df_test = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'test_data.csv'),
        index_col=0, parse_dates=True
    )
    print(f"  [OK] Train/Test data loaded")
except Exception as e:
    print(f"  [WARN] Could not load train/test: {e}")
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

# Load feature importance
try:
    df_feat_imp = pd.read_csv(os.path.join(PROCESSED_DIR, 'feature_importance.csv'))
    print(f"  [OK] Feature importance loaded")
except Exception:
    df_feat_imp = pd.DataFrame()

# Pre-compute analysis
print("[INFO] Computing analysis...")
analysis_cache = {}
if not df_hourly.empty:
    analysis_cache = get_full_analysis(df_hourly)
    print("  [OK] Analysis complete")

# ─── Load Prediction Model ────────────────────────────────────────────────────
print("[INFO] Loading prediction model...")
lr_model = None
lr_feature_order = None
device_averages = {}
global_avgs = {}

try:
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
    print(f"  [OK] Linear Regression model loaded")

    # Derive the EXACT feature order by replicating baseline_model.py prepare_data() logic
    if not df_features.empty:
        target_col = 'Global_active_power'
        exclude_cols = [target_col, 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        leaky_patterns = ['_lag_', '_rolling_', '_diff_', '_zscore', '_pct_change']

        all_features = [col for col in df_features.columns if col not in exclude_cols]
        safe_features = []
        for col in all_features:
            is_leaky = any(pattern in col for pattern in leaky_patterns)
            if target_col in col:
                is_leaky = True
            if not is_leaky:
                safe_features.append(col)

        numeric_features = df_features[safe_features].select_dtypes(include=[np.number]).columns.tolist()
        lr_feature_order = numeric_features
        print(f"  [OK] Feature order: {len(lr_feature_order)} features")
        print(f"       First 5: {lr_feature_order[:5]}")
    else:
        print("  [WARN] No feature data to derive feature order")
except Exception as e:
    print(f"  [WARN] Could not load LR model: {e}")

# Pre-compute historical feature averages per (hour, month) for realistic predictions
# CRITICAL: Must include Voltage, Global_intensity — they are the model's top features
if not df_features.empty:
    try:
        # Include ALL features the model needs, not just device features
        avg_cols = ['Voltage', 'Global_intensity',
                    'total_submetering', 'kitchen_ratio', 'laundry_ratio',
                    'hvac_ratio', 'dominant_device']
        available = [c for c in avg_cols if c in df_features.columns]
        if available and 'hour' in df_features.columns and 'month' in df_features.columns:
            grouped = df_features.groupby(['hour', 'month'])[available].mean()
            device_averages = grouped.to_dict('index')
            print(f"  [OK] Feature averages computed ({len(device_averages)} groups)")
            # Also compute global averages as fallback
            global_avgs = df_features[available].mean().to_dict()
            print(f"       Includes: {available}")
    except Exception as e:
        print(f"  [WARN] Could not compute feature averages: {e}")
        global_avgs = {}

# ─── Helper Functions ─────────────────────────────────────────────────────────

def safe_json(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def df_to_chart_data(df, columns, max_points=500):
    """Convert DataFrame to Chart.js-friendly format."""
    if df.empty:
        return {'labels': [], 'datasets': []}

    # Downsample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]

    labels = [idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
              for idx in df.index]

    datasets = []
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for i, col in enumerate(columns):
        if col in df.columns:
            datasets.append({
                'label': col.replace('_', ' ').title(),
                'data': [round(float(v), 4) if pd.notna(v) else None for v in df[col].values],
                'borderColor': colors[i % len(colors)],
                'backgroundColor': colors[i % len(colors)] + '33',
                'fill': False,
                'tension': 0.3,
                'pointRadius': 0,
            })

    return {'labels': labels, 'datasets': datasets}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Serve the EnergyPulse prediction page."""
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Live energy prediction using the trained Linear Regression model."""
    if lr_model is None or lr_feature_order is None:
        return jsonify({'error': 'Prediction model not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    try:
        hour = int(data.get('hour', 12))
        day = int(data.get('day', 15))
        month = int(data.get('month', 6))

        # Derive time features
        import datetime as dt
        # Use year from training data range (2006–2010) for consistent predictions
        year = 2008
        try:
            sample_date = dt.date(year, month, min(day, 28))
        except ValueError:
            sample_date = dt.date(year, month, 28)
        dayofweek = sample_date.weekday()
        quarter = (month - 1) // 3 + 1
        is_weekend = 1 if dayofweek >= 5 else 0
        week_of_year = sample_date.isocalendar()[1]
        day_of_month = day
        day_of_year = sample_date.timetuple().tm_yday
        is_month_start = 1 if day == 1 else 0
        is_month_end = 1 if day >= 28 else 0

        # Cyclical encodings
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        dayofweek_sin = math.sin(2 * math.pi * dayofweek / 7)
        dayofweek_cos = math.cos(2 * math.pi * dayofweek / 7)

        # Features from historical averages (Voltage, Global_intensity are CRITICAL)
        key = (hour, month)
        avg = device_averages.get(key, {})
        # Use global averages as fallback if specific (hour, month) not found
        fallback = global_avgs
        voltage = avg.get('Voltage', fallback.get('Voltage', 241.0))
        global_intensity = avg.get('Global_intensity', fallback.get('Global_intensity', 4.6))
        total_submetering = avg.get('total_submetering', fallback.get('total_submetering', 7.5))
        kitchen_ratio = avg.get('kitchen_ratio', fallback.get('kitchen_ratio', 0.15))
        laundry_ratio = avg.get('laundry_ratio', fallback.get('laundry_ratio', 0.25))
        hvac_ratio = avg.get('hvac_ratio', fallback.get('hvac_ratio', 0.60))
        dominant_device = avg.get('dominant_device', fallback.get('dominant_device', 2))

        # Build feature vector in the correct order
        feature_map = {
            'Voltage': voltage, 'Global_intensity': global_intensity,
            'hour': hour, 'day': day, 'month': month,
            'dayofweek': dayofweek, 'quarter': quarter,
            'is_weekend': is_weekend, 'year': year,
            'week_of_year': week_of_year, 'day_of_month': day_of_month,
            'day_of_year': day_of_year, 'is_month_start': is_month_start,
            'is_month_end': is_month_end,
            'hour_sin': hour_sin, 'hour_cos': hour_cos,
            'month_sin': month_sin, 'month_cos': month_cos,
            'dayofweek_sin': dayofweek_sin, 'dayofweek_cos': dayofweek_cos,
            'total_submetering': total_submetering,
            'kitchen_ratio': kitchen_ratio, 'laundry_ratio': laundry_ratio,
            'hvac_ratio': hvac_ratio, 'dominant_device': dominant_device,
        }

        X = np.array([[feature_map.get(f, 0) for f in lr_feature_order]])
        predicted_power = float(lr_model.predict(X)[0])
        predicted_power = max(0.01, predicted_power)  # clamp to positive

        # Cost estimation (Indian rates ~₹8/kWh)
        rate_per_kwh = 8.0
        daily_kwh = predicted_power * 24
        monthly_kwh = daily_kwh * 30
        daily_cost = daily_kwh * rate_per_kwh
        monthly_cost = monthly_kwh * rate_per_kwh

        # Historical average for this hour (computed from actual data)
        hist_avg = 0.0
        if not df_hourly.empty and 'Global_active_power' in df_hourly.columns:
            global_mean = float(df_hourly['Global_active_power'].mean())
            hist_avg = global_mean  # default to overall mean
            hourly_avgs = df_hourly['Global_active_power'].groupby(df_hourly.index.hour).mean()
            if hour in hourly_avgs.index:
                hist_avg = float(hourly_avgs[hour])

        # Comparison to average
        pct_diff = ((predicted_power - hist_avg) / hist_avg) * 100 if hist_avg > 0 else 0

        # Energy-saving tip based on context
        tips = [
            "Consider using energy-efficient LED lighting during evening hours.",
            "Run heavy appliances during off-peak hours (10 PM – 6 AM) for lower rates.",
            "Use smart power strips to eliminate standby power waste.",
            "Set AC thermostat 1°C higher to save ~6% on cooling costs.",
            "Maximize natural daylight during daytime to reduce lighting costs.",
            "Schedule laundry loads for cooler hours to reduce HVAC strain.",
            "Unplug chargers and devices when not in use to stop phantom loads.",
            "Use ceiling fans alongside AC to feel cooler at higher thermostat settings.",
        ]
        tip_idx = (hour + month + day) % len(tips)
        tip = tips[tip_idx]

        return jsonify({
            'predicted_power': round(predicted_power, 4),
            'daily_kwh': round(daily_kwh, 2),
            'monthly_kwh': round(monthly_kwh, 2),
            'daily_cost': round(daily_cost, 2),
            'monthly_cost': round(monthly_cost, 2),
            'currency': '₹',
            'historical_avg': round(hist_avg, 4),
            'pct_diff': round(pct_diff, 1),
            'tip': tip,
            'input': {'hour': hour, 'day': day, 'month': month,
                      'dayofweek': dayofweek, 'is_weekend': is_weekend},
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/overview')
def api_overview():
    """Get overview statistics."""
    if df_hourly.empty:
        return jsonify({'error': 'No data loaded'}), 500

    summary = analysis_cache.get('summary', get_consumption_summary(df_hourly))
    costs = analysis_cache.get('costs', estimate_costs(df_hourly))

    # LSTM accuracy — computed dynamically from predictions
    lstm_accuracy = 0.0
    if not df_predictions.empty and 'Actual' in df_predictions.columns and 'Predicted' in df_predictions.columns:
        _actual = df_predictions['Actual'].dropna()
        _predicted = df_predictions['Predicted'].dropna()
        _cidx = _actual.index.intersection(_predicted.index)
        if len(_cidx) > 0:
            _actual = _actual.loc[_cidx]
            _predicted = _predicted.loc[_cidx]
            _mask = _actual > 0.1
            if _mask.sum() > 0:
                _mape = float(np.mean(np.abs((_actual[_mask] - _predicted[_mask]) / _actual[_mask])) * 100)
                lstm_accuracy = round(100 - _mape, 1)

    overview = {
        'total_records': summary.get('total_records', len(df_hourly)),
        'date_range': summary.get('date_range', {}),
        'avg_power_kw': summary.get('global_power', {}).get('mean', 0),
        'max_power_kw': summary.get('global_power', {}).get('max', 0),
        'devices_tracked': 3,
        'features_engineered': len(df_features.columns) if not df_features.empty else 0,
        'lstm_accuracy': lstm_accuracy,
        'monthly_cost': costs.get('monthly_cost', 0),
        'currency': costs.get('currency', '₹'),
        'model_status': 'Active',
    }
    return jsonify(overview)


@app.route('/api/device-consumption')
def api_device_consumption():
    """Get device-level consumption data for charting."""
    period = request.args.get('period', 'hourly')
    device_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    if period == 'daily':
        df_src = df_daily if not df_daily.empty else df_hourly.resample('D').mean()
    elif period == 'weekly':
        df_src = df_hourly.resample('W').mean() if not df_hourly.empty else pd.DataFrame()
    elif period == 'monthly':
        df_src = df_hourly.resample('ME').mean() if not df_hourly.empty else pd.DataFrame()
    else:
        df_src = df_hourly

    chart_data = df_to_chart_data(df_src, device_cols, max_points=300)

    # Rename labels for readability
    name_map = {'Sub Metering 1': 'Kitchen', 'Sub Metering 2': 'Laundry', 'Sub Metering 3': 'HVAC'}
    for ds in chart_data['datasets']:
        ds['label'] = name_map.get(ds['label'], ds['label'])

    # Device stats
    device_stats = analysis_cache.get('device_stats', analyze_device_consumption(df_hourly))
    stats_list = []
    for key, val in device_stats.items():
        stats_list.append({
            'name': val['name'],
            'icon': val['icon'],
            'color': val['color'],
            'share': val['share_pct'],
            'mean': val['mean'],
            'max': val['max'],
            'peak_hour': val.get('peak_hour', 'N/A'),
        })

    return jsonify({
        'chart': chart_data,
        'stats': stats_list,
        'period': period
    })


@app.route('/api/consumption-trend')
def api_consumption_trend():
    """Get global active power trend for charting."""
    period = request.args.get('period', 'hourly')

    if period == 'daily':
        df_src = df_daily if not df_daily.empty else df_hourly.resample('D').mean()
    elif period == 'weekly':
        df_src = df_hourly.resample('W').mean() if not df_hourly.empty else pd.DataFrame()
    elif period == 'monthly':
        df_src = df_hourly.resample('ME').mean() if not df_hourly.empty else pd.DataFrame()
    else:
        df_src = df_hourly

    chart_data = df_to_chart_data(df_src, ['Global_active_power'], max_points=500)

    # Add hourly average pattern
    hourly_pattern = {}
    if not df_hourly.empty and 'Global_active_power' in df_hourly.columns:
        hourly_avg = df_hourly['Global_active_power'].groupby(df_hourly.index.hour).mean()
        hourly_pattern = {
            'labels': [f"{h}:00" for h in range(24)],
            'values': [round(float(v), 4) for v in hourly_avg.values]
        }

    return jsonify({
        'chart': chart_data,
        'hourly_pattern': hourly_pattern,
        'period': period
    })


@app.route('/api/predictions')
def api_predictions():
    """Get prediction results for all models."""
    if df_predictions.empty:
        return jsonify({'error': 'No prediction data available'}), 404

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Prepare LSTM chart data
    actual_col = 'Actual' if 'Actual' in df_predictions.columns else df_predictions.columns[0]
    pred_col = 'Predicted' if 'Predicted' in df_predictions.columns else df_predictions.columns[1] if len(df_predictions.columns) > 1 else None

    columns = [actual_col]
    if pred_col:
        columns.append(pred_col)

    chart_data = df_to_chart_data(df_predictions, columns, max_points=400)

    # --- Compute metrics for ALL 3 models ---

    # Helper: compute metrics from y_true and y_pred
    def _compute_metrics(y_true, y_pred):
        mae = round(float(mean_absolute_error(y_true, y_pred)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
        r2 = round(float(r2_score(y_true, y_pred)), 4)
        mask = y_true > 0.1
        mape = round(float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100), 2) if mask.sum() > 0 else 0.0
        accuracy = round(100 - mape, 1)
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'accuracy': accuracy}

    # Prepare test data for LR and XGBoost
    all_models = []
    try:
        _df = pd.read_csv(os.path.join(PROCESSED_DIR, 'data_features.csv'), index_col=0, parse_dates=True)
        _target = 'Global_active_power'
        _excl = [_target, 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        _leaky = ['_lag_', '_rolling_', '_diff_', '_zscore', '_pct_change']
        _safe = [c for c in _df.columns if c not in _excl and not any(p in c for p in _leaky) and _target not in c]
        _X = _df[_safe].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
        _y = _df[_target]
        _ts, _vs = int(0.7*len(_df)), int(0.15*len(_df))
        _Xt, _yt = _X.iloc[_ts+_vs:], _y.iloc[_ts+_vs:]

        import joblib as _jl

        # Linear Regression
        try:
            _lr = _jl.load(os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
            _yp_lr = _lr.predict(_Xt)
            lr_m = _compute_metrics(_yt.values, _yp_lr)
            lr_m['name'] = 'Linear Regression'
            lr_m['color'] = '#ef4444'
            lr_m['icon'] = '📐'
            all_models.append(lr_m)
        except Exception:
            pass

        # XGBoost
        try:
            _xgb = _jl.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
            _yp_xgb = _xgb.predict(_Xt)
            xgb_m = _compute_metrics(_yt.values, _yp_xgb)
            xgb_m['name'] = 'XGBoost'
            xgb_m['color'] = '#f59e0b'
            xgb_m['icon'] = '🌲'
            all_models.append(xgb_m)
        except Exception:
            pass
    except Exception:
        pass

    # LSTM
    lstm_m = {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'mape': 0.0, 'accuracy': 0.0,
              'name': 'LSTM', 'color': '#10b981', 'icon': '🧠'}
    if pred_col and actual_col:
        _actual = df_predictions[actual_col].dropna()
        _predicted = df_predictions[pred_col].dropna()
        _cidx = _actual.index.intersection(_predicted.index)
        if len(_cidx) > 0:
            _actual = _actual.loc[_cidx]
            _predicted = _predicted.loc[_cidx]
            lstm_m = _compute_metrics(_actual.values, _predicted.values)
            lstm_m['name'] = 'LSTM'
            lstm_m['color'] = '#10b981'
            lstm_m['icon'] = '🧠'
    all_models.append(lstm_m)

    # Best model metrics (for backward compat)
    best = max(all_models, key=lambda m: m['r2']) if all_models else lstm_m
    metrics = {**best, 'total_predictions': len(df_predictions)}

    # Error distribution (LSTM)
    error_dist = {}
    if pred_col and actual_col:
        errors = (df_predictions[actual_col] - df_predictions[pred_col]).dropna()
        error_dist = {
            'labels': [f"{round(float(v), 4)}" for v in np.histogram(errors, bins=30)[1][:-1]],
            'values': [int(v) for v in np.histogram(errors, bins=30)[0]]
        }

    return jsonify({
        'chart': chart_data,
        'metrics': metrics,
        'all_models': all_models,
        'error_distribution': error_dist
    })


@app.route('/api/model-comparison')
def api_model_comparison():
    """Compare all models: Linear Regression, XGBoost, and LSTM."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Helper: prepare test data for baseline models
    _df = None
    _Xt = None
    _yt = None
    try:
        _df = pd.read_csv(os.path.join(PROCESSED_DIR, 'data_features.csv'), index_col=0, parse_dates=True)
        _target = 'Global_active_power'
        _excl = [_target, 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        _leaky = ['_lag_', '_rolling_', '_diff_', '_zscore', '_pct_change']
        _safe = [c for c in _df.columns if c not in _excl and not any(p in c for p in _leaky) and _target not in c]
        _X = _df[_safe].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
        _y = _df[_target]
        _ts, _vs = int(0.7*len(_df)), int(0.15*len(_df))
        _Xt, _yt = _X.iloc[_ts+_vs:], _y.iloc[_ts+_vs:]
    except Exception as e:
        print(f"[WARN] Could not load test data: {e}")

    # 1. Linear Regression metrics
    baseline_metrics = {
        'name': 'Linear Regression', 'type': 'Baseline',
        'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'mape': 0.0,
        'training_time': '< 1 min', 'color': '#ef4444'
    }
    try:
        import joblib as _jl
        _lr = _jl.load(os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
        if _Xt is not None:
            _yp = _lr.predict(_Xt)
            baseline_metrics['mae'] = round(float(mean_absolute_error(_yt, _yp)), 4)
            baseline_metrics['rmse'] = round(float(np.sqrt(mean_squared_error(_yt, _yp))), 4)
            baseline_metrics['r2'] = round(float(r2_score(_yt, _yp)), 4)
            _m = _yt > 0.1
            if _m.sum() > 0:
                baseline_metrics['mape'] = round(float(np.mean(np.abs((_yt[_m] - _yp[_m]) / _yt[_m])) * 100), 2)
    except Exception as e:
        print(f"[WARN] LR metrics error: {e}")

    # 2. XGBoost metrics
    xgb_metrics = {
        'name': 'XGBoost', 'type': 'Ensemble',
        'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'mape': 0.0,
        'training_time': '~2 min', 'color': '#f59e0b'
    }
    try:
        import joblib as _jl
        _xgb = _jl.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
        if _Xt is not None:
            _yp_x = _xgb.predict(_Xt)
            xgb_metrics['mae'] = round(float(mean_absolute_error(_yt, _yp_x)), 4)
            xgb_metrics['rmse'] = round(float(np.sqrt(mean_squared_error(_yt, _yp_x))), 4)
            xgb_metrics['r2'] = round(float(r2_score(_yt, _yp_x)), 4)
            _m = _yt > 0.1
            if _m.sum() > 0:
                xgb_metrics['mape'] = round(float(np.mean(np.abs((_yt[_m] - _yp_x[_m]) / _yt[_m])) * 100), 2)
    except Exception as e:
        print(f"[WARN] XGBoost metrics error: {e}")

    # 3. LSTM metrics — from saved predictions
    lstm_metrics = {
        'name': 'LSTM Neural Network', 'type': 'Deep Learning',
        'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'mape': 0.0,
        'training_time': '~15 min', 'color': '#10b981'
    }
    if not df_predictions.empty and 'Actual' in df_predictions.columns and 'Predicted' in df_predictions.columns:
        actual = df_predictions['Actual'].dropna()
        predicted = df_predictions['Predicted'].dropna()
        common_idx = actual.index.intersection(predicted.index)
        if len(common_idx) > 0:
            actual = actual.loc[common_idx]
            predicted = predicted.loc[common_idx]
            lstm_metrics['mae'] = round(float(mean_absolute_error(actual, predicted)), 4)
            lstm_metrics['rmse'] = round(float(np.sqrt(mean_squared_error(actual, predicted))), 4)
            lstm_metrics['r2'] = round(float(r2_score(actual, predicted)), 4)
            mask = actual > 0.1
            if mask.sum() > 0:
                lstm_metrics['mape'] = round(float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100), 2)

    # Feature importance
    feat_imp = []
    if not df_feat_imp.empty:
        top_features = df_feat_imp.head(10)
        feat_imp = top_features.to_dict('records')

    # Determine winner from all 3 models
    all_models = {'Linear Regression': baseline_metrics, 'XGBoost': xgb_metrics, 'LSTM': lstm_metrics}
    winner = max(all_models, key=lambda k: all_models[k]['r2'])

    return jsonify({
        'baseline': baseline_metrics,
        'xgboost': xgb_metrics,
        'lstm': lstm_metrics,
        'feature_importance': feat_imp,
        'winner': winner
    })


@app.route('/api/suggestions')
def api_suggestions():
    """Get smart energy-saving suggestions."""
    suggestions = analysis_cache.get('suggestions', [])
    if not suggestions and not df_hourly.empty:
        suggestions = generate_suggestions(df_hourly)

    costs = analysis_cache.get('costs', {})
    anomaly_count = len(analysis_cache.get('anomalies', []))

    return jsonify({
        'suggestions': suggestions,
        'costs': costs,
        'anomaly_count': anomaly_count
    })


@app.route('/api/anomalies')
def api_anomalies():
    """Get detected anomalies."""
    anomalies = analysis_cache.get('anomalies', [])
    if not anomalies and not df_hourly.empty:
        anomalies = detect_anomalies(df_hourly)

    return jsonify({
        'anomalies': anomalies,
        'total': len(anomalies),
        'high_severity': len([a for a in anomalies if a['severity'] == 'HIGH']),
        'medium_severity': len([a for a in anomalies if a['severity'] == 'MEDIUM'])
    })


@app.route('/api/hourly-pattern')
def api_hourly_pattern():
    """Get 24-hour consumption pattern."""
    if df_hourly.empty or 'Global_active_power' not in df_hourly.columns:
        return jsonify({'error': 'No data'}), 404

    cols = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    patterns = {}
    for col in cols:
        if col in df_hourly.columns:
            hourly_avg = df_hourly[col].groupby(df_hourly.index.hour).mean()
            label = col.replace('Sub_metering_1', 'Kitchen').replace(
                'Sub_metering_2', 'Laundry').replace(
                'Sub_metering_3', 'HVAC').replace('_', ' ').title()
            patterns[label] = [round(float(v), 4) for v in hourly_avg.values]

    return jsonify({
        'labels': [f"{h:02d}:00" for h in range(24)],
        'patterns': patterns
    })


@app.route('/visualizations/<filename>')
def serve_visualization(filename):
    """Serve milestone visualization images."""
    return send_from_directory(VIZ_DIR, filename)


@app.route('/api/visualizations')
def api_visualizations():
    """List available visualization files."""
    viz_files = []
    if os.path.exists(VIZ_DIR):
        for f in sorted(os.listdir(VIZ_DIR)):
            if f.endswith('.png'):
                milestone = 'Milestone ' + f.split('_')[0].replace('Milestone', '').strip() if 'Milestone' in f else 'Other'
                viz_files.append({
                    'filename': f,
                    'url': f'/visualizations/{f}',
                    'title': f.replace('_', ' ').replace('.png', ''),
                    'milestone': milestone
                })

    return jsonify({'visualizations': viz_files})


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SMART ENERGY DASHBOARD")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

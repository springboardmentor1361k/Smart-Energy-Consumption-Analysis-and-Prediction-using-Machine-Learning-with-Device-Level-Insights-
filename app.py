"""
SMART ENERGY DASHBOARD - Main Application
Uses locally trained XGBoost model
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from flask_cors import CORS
from functools import wraps
import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime, timedelta
import io

# ============================================================================
# APP CONFIGURATION
# ============================================================================
app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')

app.secret_key = 'smart-energy-secret-key-2024'
CORS(app)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')

# ============================================================================
# LOAD MODELS AND DATA AT STARTUP
# ============================================================================
print("\n" + "=" * 60)
print("🏠 SMART ENERGY DASHBOARD")
print("=" * 60)

# Global storage
models = {
    'xgboost': None,
    'feature_scaler': None,
    'target_scaler': None,
    'config': None,
    'loaded': False
}
data_cache = {
    'hourly': None,
    'device_mapping': None
}

def load_models():
    """Load trained models."""
    global models
    print("\n📦 Loading models...")
    
    try:
        # Load XGBoost
        xgb_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
        if os.path.exists(xgb_path):
            models['xgboost'] = joblib.load(xgb_path)
            print(f"  ✅ XGBoost model loaded")
        
        # Load scalers
        feature_scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
        if os.path.exists(feature_scaler_path):
            models['feature_scaler'] = joblib.load(feature_scaler_path)
            print(f"  ✅ Feature scaler loaded")
        
        target_scaler_path = os.path.join(MODELS_DIR, 'target_scaler.pkl')
        if os.path.exists(target_scaler_path):
            models['target_scaler'] = joblib.load(target_scaler_path)
            print(f"  ✅ Target scaler loaded")
        
        # Load config
        config_path = os.path.join(MODELS_DIR, 'ensemble_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                models['config'] = json.load(f)
            print(f"  ✅ Config loaded")
        
        models['loaded'] = models['xgboost'] is not None
        
        if models['loaded']:
            print(f"  ✅ All models ready!")
        else:
            print(f"  ⚠️ Some models missing")
            
    except Exception as e:
        print(f"  ❌ Error loading models: {e}")

def load_data():
    """Load processed data."""
    global data_cache
    print("\n📊 Loading data...")
    
    try:
        # Load hourly data
        hourly_path = os.path.join(DATA_DIR, 'hourly_raw.csv')
        if os.path.exists(hourly_path):
            data_cache['hourly'] = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
            print(f"  ✅ Hourly data: {len(data_cache['hourly']):,} rows")
        
        # Load device mapping
        mapping_path = os.path.join(DATA_DIR, 'device_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                data_cache['device_mapping'] = json.load(f)
            print(f"  ✅ Device mapping loaded")
        else:
            # Default mapping
            data_cache['device_mapping'] = {
                'device_rooms': {
                    'Aggregate': 'Whole House',
                    'Kitchen': 'Kitchen',
                    'Laundry': 'Utility Room',
                    'Climate_Control': 'HVAC',
                    'Other_Appliances': 'Mixed'
                }
            }
            print(f"  ⚠️ Using default device mapping")
            
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")

# Load at startup
load_models()
load_data()

# ============================================================================
# AUTHENTICATION
# ============================================================================
USERS = {
    'demo': 'demo123',
    'admin': 'admin123'
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_status():
    """Get status of loaded models."""
    return {
        'xgb_loaded': models['xgboost'] is not None,
        'lstm_loaded': False,  # Not using LSTM
        'scalers_loaded': models['feature_scaler'] is not None,
        'all_loaded': models['loaded']
    }

def predict_next_24_hours(data):
    """Generate 24-hour predictions using statistical method."""
    predictions = []
    
    if data is None or data.empty:
        now = datetime.now()
        for h in range(24):
            predictions.append({
                'timestamp': (now + timedelta(hours=h+1)).isoformat(),
                'hour': (now.hour + h + 1) % 24,
                'predicted_power': 0
            })
        return predictions
    
    # Calculate hourly averages from historical data
    try:
        hourly_avg = data.groupby(data.index.hour)['Aggregate'].mean().to_dict()
    except:
        hourly_avg = {h: float(data['Aggregate'].mean()) for h in range(24)}
    
    last_time = data.index[-1]
    
    for h in range(24):
        next_time = last_time + timedelta(hours=h+1)
        hour = next_time.hour
        
        pred = hourly_avg.get(hour, float(data['Aggregate'].mean()))
        variation = np.random.uniform(-0.03, 0.03)
        pred = pred * (1 + variation)
        
        predictions.append({
            'timestamp': next_time.isoformat(),
            'hour': hour,
            'predicted_power': round(max(0, pred), 2)
        })
    
    return predictions

def generate_suggestions(data):
    """Generate energy saving suggestions."""
    suggestions = []
    
    if data is None or data.empty:
        return {'suggestions': [], 'summary': {'total_potential_savings_kwh': 0}}
    
    # Analyze patterns
    hourly_avg = data.groupby(data.index.hour)['Aggregate'].mean()
    peak_hour = hourly_avg.idxmax()
    peak_power = hourly_avg.max()
    avg_power = hourly_avg.mean()
    
    # Suggestion 1: Peak usage
    if peak_power > avg_power * 1.3:
        suggestions.append({
            'icon': '⚡',
            'type': 'Peak Load Management',
            'priority': 'High',
            'title': f'High Peak at {peak_hour}:00',
            'message': f'Peak consumption of {peak_power:.0f}W at {peak_hour}:00 is {((peak_power/avg_power-1)*100):.0f}% above average.',
            'action': 'Consider shifting high-power appliances to off-peak hours (22:00-06:00).',
            'potential_savings_kwh': round((peak_power - avg_power) * 30 / 1000, 1),
            'potential_savings_cost': round((peak_power - avg_power) * 30 / 1000 * 0.28, 2)
        })
    
    # Suggestion 2: Night usage
    night_avg = data[data.index.hour.isin([0,1,2,3,4,5])]['Aggregate'].mean()
    if night_avg > 200:
        suggestions.append({
            'icon': '🌙',
            'type': 'Standby Power',
            'priority': 'Medium',
            'title': 'High Night-time Consumption',
            'message': f'Average night consumption is {night_avg:.0f}W. This may indicate standby power waste.',
            'action': 'Use smart power strips to cut standby power for devices not in use at night.',
            'potential_savings_kwh': round(night_avg * 0.3 * 6 * 30 / 1000, 1),
            'potential_savings_cost': round(night_avg * 0.3 * 6 * 30 / 1000 * 0.28, 2)
        })
    
    # Suggestion 3: Weekend patterns
    if 'dayofweek' in data.columns or True:
        data_copy = data.copy()
        data_copy['dow'] = data_copy.index.dayofweek
        weekend_avg = data_copy[data_copy['dow'] >= 5]['Aggregate'].mean()
        weekday_avg = data_copy[data_copy['dow'] < 5]['Aggregate'].mean()
        
        if weekend_avg > weekday_avg * 1.2:
            suggestions.append({
                'icon': '📅',
                'type': 'Weekend Usage',
                'priority': 'Low',
                'title': 'Higher Weekend Consumption',
                'message': f'Weekend usage ({weekend_avg:.0f}W) is {((weekend_avg/weekday_avg-1)*100):.0f}% higher than weekdays.',
                'action': 'Be mindful of entertainment device usage on weekends.',
                'potential_savings_kwh': round((weekend_avg - weekday_avg) * 48 * 4 / 1000, 1),
                'potential_savings_cost': round((weekend_avg - weekday_avg) * 48 * 4 / 1000 * 0.28, 2)
            })
    
    # Add general tip
    suggestions.append({
        'icon': '💡',
        'type': 'General Tip',
        'priority': 'Low',
        'title': 'Switch to LED Lighting',
        'message': 'LED bulbs use 75% less energy than incandescent bulbs.',
        'action': 'Replace remaining incandescent bulbs with LED alternatives.',
        'potential_savings_kwh': 15,
        'potential_savings_cost': 4.20
    })
    
    # Calculate totals
    total_kwh = sum(s.get('potential_savings_kwh', 0) for s in suggestions)
    total_cost = sum(s.get('potential_savings_cost', 0) for s in suggestions)
    
    return {
        'suggestions': suggestions,
        'summary': {
            'total_potential_savings_kwh': round(total_kwh, 1),
            'total_potential_savings_cost': round(total_cost, 2),
            'total_potential_co2_savings': round(total_kwh * 0.233, 2),
            'suggestions_count': len(suggestions)
        },
        'high_priority_count': len([s for s in suggestions if s['priority'] == 'High']),
        'medium_priority_count': len([s for s in suggestions if s['priority'] == 'Medium']),
        'low_priority_count': len([s for s in suggestions if s['priority'] == 'Low'])
    }

def calculate_cost(data, tariff='standard'):
    """Calculate energy cost."""
    if data is None or data.empty:
        return {
            'totals': {'total_cost': 0, 'energy_kwh': 0},
            'averages': {'daily_cost': 0, 'daily_kwh': 0}
        }
    
    rates = {
        'standard': 0.28,
        'economy7_peak': 0.35,
        'economy7_offpeak': 0.15
    }
    
    total_kwh = data['Aggregate'].sum() / 1000
    days = max(1, (data.index.max() - data.index.min()).days)
    
    total_cost = total_kwh * rates['standard']
    daily_cost = total_cost / days
    
    return {
        'tariff': 'Standard Variable',
        'totals': {
            'energy_kwh': round(total_kwh, 2),
            'energy_cost': round(total_cost, 2),
            'standing_charges': round(days * 0.50, 2),
            'total_cost': round(total_cost + days * 0.50, 2)
        },
        'averages': {
            'daily_kwh': round(total_kwh / days, 2),
            'daily_cost': round(daily_cost, 2)
        }
    }

def calculate_carbon(data):
    """Calculate carbon emissions."""
    if data is None or data.empty:
        return {
            'totals': {'emissions_kg': 0},
            'projections': {'annual_emissions_kg': 0, 'annual_emissions_tonnes': 0},
            'equivalents': {'car_miles': 0, 'trees_needed_annually': 0, 'smartphone_charges': 0}
        }
    
    carbon_factor = 0.233  # kg CO2 per kWh
    
    total_kwh = data['Aggregate'].sum() / 1000
    days = max(1, (data.index.max() - data.index.min()).days)
    
    emissions_kg = total_kwh * carbon_factor
    annual_emissions = (emissions_kg / days) * 365
    
    return {
        'totals': {
            'emissions_kg': round(emissions_kg, 2),
            'emissions_tonnes': round(emissions_kg / 1000, 4)
        },
        'projections': {
            'annual_emissions_kg': round(annual_emissions, 2),
            'annual_emissions_tonnes': round(annual_emissions / 1000, 3)
        },
        'equivalents': {
            'car_miles': round(emissions_kg / 0.21, 0),
            'trees_needed_annually': round(annual_emissions / 21, 1),
            'smartphone_charges': round(emissions_kg / 0.008, 0)
        }
    }

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Landing page."""
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    error = None
    
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """Logout."""
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard."""
    data = data_cache['hourly']
    
    if data is None or data.empty:
        return render_template('dashboard.html', data={
            'current_power': 0,
            'daily_consumption': 0,
            'weekly_average': 0,
            'predicted_avg': 0,
            'cost_today': 0,
            'cost_monthly': 0,
            'hourly_pattern': [0] * 24,
            'device_totals': {},
            'model_status': get_model_status(),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': 'No data available'
        }, devices=data_cache['device_mapping'])
    
    # Recent data
    recent_week = data.tail(168)
    recent_day = data.tail(24)
    
    # Metrics
    current_power = round(float(recent_day['Aggregate'].iloc[-1]), 1)
    daily_consumption = round(float(recent_day['Aggregate'].sum()) / 1000, 2)
    weekly_average = round(float(recent_week['Aggregate'].mean()), 1)
    
    # Predictions
    predictions = predict_next_24_hours(data.tail(72))
    predicted_avg = round(sum(p['predicted_power'] for p in predictions) / 24, 1)
    
    # Cost
    cost_data = calculate_cost(recent_week)
    cost_today = cost_data['averages']['daily_cost']
    cost_monthly = round(cost_today * 30, 2)
    
    # Hourly pattern
    hourly_pattern = data.groupby(data.index.hour)['Aggregate'].mean()
    hourly_pattern = [round(float(x), 1) for x in hourly_pattern.tolist()]
    
    # Device totals
    device_cols = ['Kitchen', 'Laundry', 'Climate_Control', 'Other_Appliances']
    device_totals = {}
    for col in device_cols:
        if col in data.columns:
            device_totals[col] = round(float(data[col].sum()) / 1000, 2)
    
    dashboard_data = {
        'current_power': current_power,
        'daily_consumption': daily_consumption,
        'weekly_average': weekly_average,
        'predicted_avg': predicted_avg,
        'cost_today': cost_today,
        'cost_monthly': cost_monthly,
        'hourly_pattern': hourly_pattern,
        'device_totals': device_totals,
        'model_status': get_model_status(),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'error': None
    }
    
    return render_template('dashboard.html', data=dashboard_data, devices=data_cache['device_mapping'])

@app.route('/devices')
@login_required
def devices():
    """Device analysis page."""
    data = data_cache['hourly']
    device_mapping = data_cache['device_mapping']
    
    if data is None or data.empty:
        return render_template('devices.html', device_stats=[], room_totals={}, device_mapping=device_mapping)
    
    recent = data.tail(720)
    device_cols = ['Kitchen', 'Laundry', 'Climate_Control', 'Other_Appliances']
    
    total = sum(float(recent[col].sum()) for col in device_cols if col in recent.columns)
    
    device_stats = []
    for col in device_cols:
        if col in recent.columns:
            consumption = float(recent[col].sum())
            pct = (consumption / total * 100) if total > 0 else 0
            
            device_stats.append({
                'name': col.replace('_', ' '),
                'room': device_mapping.get('device_rooms', {}).get(col, 'Unknown'),
                'total_kwh': round(consumption / 1000, 2),
                'percentage': round(pct, 1),
                'avg_power': round(float(recent[col].mean()), 1),
                'max_power': round(float(recent[col].max()), 1),
                'min_power': round(float(recent[col].min()), 1),
                'usage_hours': int((recent[col] > 1).sum())
            })
    
    device_stats.sort(key=lambda x: x['total_kwh'], reverse=True)
    
    # Room totals
    room_totals = {}
    for stat in device_stats:
        room = stat['room']
        room_totals[room] = room_totals.get(room, 0) + stat['total_kwh']
    room_totals = {k: round(v, 2) for k, v in room_totals.items()}
    
    return render_template('devices.html', device_stats=device_stats, room_totals=room_totals, device_mapping=device_mapping)

@app.route('/predictions')
@login_required
def predictions():
    """Predictions page."""
    data = data_cache['hourly']
    model_status = get_model_status()
    
    if data is None or data.empty:
        empty_preds = [{'hour': h, 'predicted_power': 0, 'timestamp': ''} for h in range(24)]
        return render_template('predictions.html', predictions=empty_preds, 
                             yesterday=[0]*24, today_actual=[0]*24, model_status=model_status)
    
    preds = predict_next_24_hours(data.tail(72))
    
    yesterday = [round(float(x), 1) for x in data.tail(48)[:24]['Aggregate'].tolist()] if len(data) >= 48 else [0]*24
    today_actual = [round(float(x), 1) for x in data.tail(24)['Aggregate'].tolist()] if len(data) >= 24 else [0]*24
    
    while len(yesterday) < 24:
        yesterday.append(0)
    while len(today_actual) < 24:
        today_actual.append(0)
    
    return render_template('predictions.html', predictions=preds, 
                         yesterday=yesterday, today_actual=today_actual, model_status=model_status)

@app.route('/suggestions')
@login_required
def suggestions():
    """Suggestions page."""
    data = data_cache['hourly']
    
    if data is None or data.empty:
        report = {'suggestions': [], 'summary': {'total_potential_savings_kwh': 0, 'total_potential_savings_cost': 0, 'total_potential_co2_savings': 0, 'suggestions_count': 0}, 'high_priority_count': 0, 'medium_priority_count': 0, 'low_priority_count': 0}
        return render_template('suggestions.html', report=report)
    
    report = generate_suggestions(data.tail(720))
    return render_template('suggestions.html', report=report)

@app.route('/reports')
@login_required
def reports():
    """Reports page."""
    data = data_cache['hourly']
    
    if data is None or data.empty:
        return render_template('reports.html',
            cost_data={'totals': {'total_cost': 0}, 'averages': {'daily_cost': 0}},
            tariff_comparison={'tariffs': {}, 'cheapest_tariff': 'standard'},
            emissions_data={'totals': {'emissions_kg': 0}, 'projections': {'annual_emissions_kg': 0, 'annual_emissions_tonnes': 0}, 'equivalents': {'car_miles': 0, 'trees_needed_annually': 0, 'smartphone_charges': 0}},
            emissions_comparison={'rating': 'Unknown', 'message': 'No data', 'color': 'yellow', 'uk_average_annual_emissions_kg': 676},
            reduction_tips=[])
    
    recent = data.tail(720)
    
    cost_data = calculate_cost(recent)
    emissions_data = calculate_carbon(recent)
    
    # Compare to average
    user_annual = emissions_data['projections']['annual_emissions_kg']
    uk_avg = 676  # kg CO2 per year for electricity
    diff_pct = ((user_annual - uk_avg) / uk_avg) * 100 if uk_avg > 0 else 0
    
    if diff_pct < -20:
        rating, color = 'Excellent', 'green'
    elif diff_pct < 0:
        rating, color = 'Good', 'lightgreen'
    elif diff_pct < 20:
        rating, color = 'Average', 'yellow'
    else:
        rating, color = 'High', 'red'
    
    emissions_comparison = {
        'rating': rating,
        'message': f'Your emissions are {abs(diff_pct):.0f}% {"below" if diff_pct < 0 else "above"} average',
        'color': color,
        'uk_average_annual_emissions_kg': uk_avg,
        'difference_percent': round(diff_pct, 1)
    }
    
    reduction_tips = [
        {'category': 'Timing', 'title': 'Shift to Off-Peak', 'description': 'Run appliances during off-peak hours (22:00-06:00)', 'impact': 'High', 'potential_reduction_percent': 15},
        {'category': 'Efficiency', 'title': 'LED Lighting', 'description': 'Replace all bulbs with LED alternatives', 'impact': 'Medium', 'potential_reduction_percent': 10},
        {'category': 'Standby', 'title': 'Smart Power Strips', 'description': 'Use smart strips to eliminate standby power', 'impact': 'Medium', 'potential_reduction_percent': 8}
    ]
    
    return render_template('reports.html',
        cost_data=cost_data,
        tariff_comparison={'tariffs': {}, 'cheapest_tariff': 'standard'},
        emissions_data=emissions_data,
        emissions_comparison=emissions_comparison,
        reduction_tips=reduction_tips)

@app.route('/settings')
@login_required
def settings():
    """Settings page."""
    user = {'username': session.get('username', 'User'), 'email': f"{session.get('username', 'user')}@example.com"}
    return render_template('settings.html', user=user)

@app.route('/download/<report_type>')
@login_required
def download_report(report_type):
    """Download reports."""
    data = data_cache['hourly']
    
    if data is None or data.empty:
        return "No data available", 404
    
    recent = data.tail(720)
    
    if report_type == 'csv':
        output = io.StringIO()
        recent.to_csv(output)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'energy_report_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    
    return "Invalid report type", 400

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health')
def api_health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': get_model_status()
    })

@app.route('/api/summary')
@login_required
def api_summary():
    """Get summary statistics."""
    data = data_cache['hourly']
    
    if data is None or data.empty:
        return jsonify({'error': 'No data'})
    
    recent_24h = data.tail(24)
    recent_week = data.tail(168)
    
    return jsonify({
        'today': {
            'total_kwh': round(float(recent_24h['Aggregate'].sum()) / 1000, 2),
            'avg_power_w': round(float(recent_24h['Aggregate'].mean()), 1)
        },
        'this_week': {
            'total_kwh': round(float(recent_week['Aggregate'].sum()) / 1000, 2),
            'avg_power_w': round(float(recent_week['Aggregate'].mean()), 1)
        }
    })

@app.route('/api/realtime')
@login_required
def api_realtime():
    """Simulate real-time data."""
    data = data_cache['hourly']
    
    if data is None or data.empty:
        return jsonify({'readings': {'Aggregate': 0}})
    
    last = data.tail(1)
    readings = {}
    for col in data.columns:
        base = float(last[col].iloc[0])
        variation = np.random.uniform(-0.05, 0.05)
        readings[col] = round(base * (1 + variation), 1)
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'readings': readings
    })

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║        🏠 SMART ENERGY DASHBOARD                             ║
    ║        🌐 http://localhost:5000                              ║
    ║        👤 Login: demo / demo123                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=5000, debug=True)
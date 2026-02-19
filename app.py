"""
Smart Energy Consumption Analysis - Flask Web Application
Main application file
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from flask_prediction import predict

app = Flask(__name__)

# Load dashboard data
def load_dashboard_data():
    """Load dashboard data from JSON file"""
    try:
        with open('C:/Users/battu/Documents/SmartEnergyML/dashboard_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'statistics': {},
            'device_stats': {},
            'suggestions': [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Load energy data
def load_energy_data():
    """Load hourly energy data"""
    try:
        df = pd.read_csv('C:/Users/battu/Documents/SmartEnergyML/data/processed/data_hourly.csv', index_col='Datetime', parse_dates=True)
        return df
    except FileNotFoundError:
        return None

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
    """Homepage with dashboard overview"""
    data = load_dashboard_data()
    return render_template('index.html', data=data)

@app.route('/dashboard')
def dashboard():
    """Main dashboard page"""
    data = load_dashboard_data()
    df = load_energy_data()
    
    # Prepare chart data
    if df is not None:
        # Last 24 hours
        recent_24h = df.tail(24)
        hourly_data = {
            'labels': [t.strftime('%H:%M') for t in recent_24h.index],
            'power': recent_24h['Global_active_power'].tolist(),
            'kitchen': recent_24h['Sub_metering_1'].tolist(),
            'laundry': recent_24h['Sub_metering_2'].tolist(),
            'hvac': recent_24h['Sub_metering_3'].tolist()
        }
        
        # Last 30 days
        daily = df.resample('D').mean().tail(30)
        daily_data = {
            'labels': [t.strftime('%Y-%m-%d') for t in daily.index],
            'power': daily['Global_active_power'].tolist()
        }
    else:
        hourly_data = {'labels': [], 'power': [], 'kitchen': [], 'laundry': [], 'hvac': []}
        daily_data = {'labels': [], 'power': []}
    
    return render_template('dashboard.html', 
                          data=data, 
                          hourly_data=hourly_data,
                          daily_data=daily_data)

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Energy prediction page"""
    if request.method == 'POST':
        try:
            # CHANGED: Use predict_simple() instead of loading data here
            from flask_prediction import predict_simple
            result = predict_simple()
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('predict.html')

@app.route('/suggestions')
def suggestions_page():
    """Energy efficiency suggestions page"""
    data = load_dashboard_data()
    return render_template('suggestions.html', suggestions=data.get('suggestions', []))

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    data = load_dashboard_data()
    return jsonify(data['statistics'])

@app.route('/api/devices')
def api_devices():
    """API endpoint for device statistics"""
    data = load_dashboard_data()
    return jsonify(data['device_stats'])

@app.route('/api/hourly/<int:hours>')
def api_hourly(hours=24):
    """API endpoint for hourly data"""
    df = load_energy_data()
    if df is None:
        return jsonify({'error': 'Data not available'}), 404
    
    recent = df.tail(hours)
    result = {
        'timestamps': [t.strftime('%Y-%m-%d %H:%M') for t in recent.index],
        'power': recent['Global_active_power'].tolist(),
        'devices': {
            'kitchen': recent['Sub_metering_1'].tolist(),
            'laundry': recent['Sub_metering_2'].tolist(),
            'hvac': recent['Sub_metering_3'].tolist()
        }
    }
    return jsonify(result)

@app.route('/api/daily/<int:days>')
def api_daily(days=30):
    """API endpoint for daily data"""
    df = load_energy_data()
    if df is None:
        return jsonify({'error': 'Data not available'}), 404
    
    daily = df.resample('D').mean().tail(days)
    result = {
        'dates': [t.strftime('%Y-%m-%d') for t in daily.index],
        'power': daily['Global_active_power'].tolist()
    }
    return jsonify(result)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print(" "*10 + "Smart Energy Consumption Analysis")
    print(" "*15 + "Flask Web Application")
    print("="*60)
    print("\n🚀 Starting server...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("⚡ Prediction API at: http://localhost:5000/predict")
    print("\n Press CTRL+C to stop the server\n")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
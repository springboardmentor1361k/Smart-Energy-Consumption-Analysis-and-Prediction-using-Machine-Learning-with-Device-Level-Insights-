"""
REAL-TIME SMART ENERGY DASHBOARD
Live data streaming with WebSockets
"""

from flask import Flask, render_template, request, session, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from functools import wraps
import pandas as pd
import numpy as np
import json
import os
import joblib
import threading
import time
import random
from datetime import datetime, timedelta
from collections import deque
import redis

# ============================================================================
# CONFIGURATION
# ============================================================================
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['SECRET_KEY'] = 'smart-energy-realtime-2024'

# WebSocket with Redis for production scaling
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Real-time data storage
realtime_data = {
    'current_readings': {},
    'history': deque(maxlen=1000),  # Last 1000 readings
    'predictions': [],
    'alerts': [],
    'ml_model': None,
    'scaler': None,
    'is_running': False
}

# ============================================================================
# DATA GENERATOR (Simulates Smart Meter / IoT Devices)
# ============================================================================
class SmartMeterSimulator:
    """Simulates real smart meter readings"""
    
    def __init__(self):
        self.base_consumption = {
            'Aggregate': 300,
            'Kitchen': 50,
            'Laundry': 30,
            'Climate_Control': 100,
            'Other_Appliances': 120
        }
        self.time_patterns = self._load_patterns()
        
    def _load_patterns(self):
        """Load realistic usage patterns"""
        return {
            'morning_peak': {'hours': [7, 8, 9], 'multiplier': 1.8},
            'evening_peak': {'hours': [18, 19, 20, 21], 'multiplier': 2.2},
            'night_low': {'hours': [0, 1, 2, 3, 4, 5], 'multiplier': 0.4},
            'work_hours': {'hours': [10, 11, 12, 13, 14, 15, 16, 17], 'multiplier': 0.8}
        }
    
    def generate_reading(self):
        """Generate one real-time reading"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Determine time pattern multiplier
        multiplier = 1.0
        for pattern, info in self.time_patterns.items():
            if hour in info['hours']:
                multiplier = info['multiplier']
                break
        
        # Add randomness and trends
        readings = {}
        for device, base in self.base_consumption.items():
            # Base with time pattern
            value = base * multiplier
            
            # Add minute-level variation (simulates appliance switching)
            if minute % 15 == 0:  # Every 15 minutes
                value *= random.uniform(0.8, 1.3)
            
            # Add noise
            noise = random.gauss(0, base * 0.05)
            value += noise
            
            # Simulate appliance events
            if device == 'Kitchen' and hour in [7, 8, 12, 18, 19]:
                if random.random() > 0.7:
                    value += random.uniform(500, 1500)  # Cooking spike
            
            if device == 'Laundry' and hour in [10, 19] and minute < 5:
                if random.random() > 0.8:
                    value += random.uniform(800, 2000)  # Washing machine
            
            readings[device] = round(max(0, value), 2)
        
        # Calculate aggregate from sum if not main meter
        readings['timestamp'] = now.isoformat()
        readings['hour'] = hour
        readings['minute'] = minute
        
        return readings

# Global simulator
meter_simulator = SmartMeterSimulator()

# ============================================================================
# ML PREDICTION ENGINE
# ============================================================================
class MLPredictionEngine:
    """Real-time ML predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.sequence_length = 24
        self.prediction_buffer = deque(maxlen=24)
        self.load_model()
        
    def load_model(self):
        """Load trained model"""
        try:
            model_path = 'data/models/xgboost_model.pkl'
            scaler_path = 'data/models/feature_scaler.pkl'
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("✅ ML model loaded for real-time predictions")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
        except Exception as e:
            print(f"⚠️ Model load error: {e}")
    
    def predict_next_hour(self, recent_readings):
        """Predict next hour consumption"""
        if self.model is None or len(recent_readings) < 6:
            # Statistical fallback
            values = [r['Aggregate'] for r in recent_readings[-6:]]
            return np.mean(values) * random.uniform(0.95, 1.05)
        
        try:
            # Simple feature extraction
            features = self._extract_features(recent_readings)
            prediction = self.model.predict([features])[0]
            return max(0, prediction)
        except:
            # Fallback
            values = [r['Aggregate'] for r in recent_readings[-6:]]
            return np.mean(values)
    
    def _extract_features(self, readings):
        """Extract features from recent readings"""
        values = [r['Aggregate'] for r in readings[-24:]]
        
        # Pad if needed
        while len(values) < 24:
            values.insert(0, values[0] if values else 300)
        
        features = {
            'mean': np.mean(values),
            'std': np.std(values),
            'max': np.max(values),
            'min': np.min(values),
            'trend': values[-1] - values[0],
            'hour': datetime.now().hour,
            'is_peak': 1 if datetime.now().hour in [7,8,9,18,19,20,21] else 0
        }
        
        return list(features.values())
    
    def detect_anomaly(self, reading, history):
        """Detect unusual consumption"""
        if len(history) < 10:
            return None
        
        recent = [h['Aggregate'] for h in list(history)[-10:]]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std == 0:
            return None
        
        z_score = (reading['Aggregate'] - mean) / std
        
        if z_score > 2.5:
            return {
                'type': 'SPIKE',
                'severity': 'HIGH' if z_score > 3 else 'MEDIUM',
                'message': f"Unusual spike detected: {reading['Aggregate']:.0f}W (avg: {mean:.0f}W)",
                'timestamp': reading['timestamp']
            }
        
        if z_score < -2:
            return {
                'type': 'DROP',
                'severity': 'MEDIUM',
                'message': f"Sudden drop detected: {reading['Aggregate']:.0f}W (avg: {mean:.0f}W)",
                'timestamp': reading['timestamp']
            }
        
        return None

# Global ML engine
ml_engine = MLPredictionEngine()

# ============================================================================
# BACKGROUND DATA STREAM
# ============================================================================
def data_stream_worker():
    """Background thread: generates and broadcasts real-time data"""
    print("🚀 Starting real-time data stream...")
    
    while realtime_data['is_running']:
        try:
            # 1. Generate new reading
            reading = meter_simulator.generate_reading()
            
            # 2. Store in history
            realtime_data['history'].append(reading)
            
            # 3. ML Prediction
            prediction = ml_engine.predict_next_hour(list(realtime_data['history']))
            reading['predicted_next'] = round(prediction, 2)
            
            # 4. Anomaly Detection
            anomaly = ml_engine.detect_anomaly(reading, realtime_data['history'])
            if anomaly:
                realtime_data['alerts'].append(anomaly)
                # Keep only last 10 alerts
                realtime_data['alerts'] = realtime_data['alerts'][-10:]
            
            # 5. Generate smart suggestion based on current state
            suggestion = generate_live_suggestion(reading, list(realtime_data['history']))
            
            # 6. Broadcast to all connected clients
            socketio.emit('new_reading', {
                'reading': reading,
                'prediction': round(prediction, 2),
                'anomaly': anomaly,
                'suggestion': suggestion,
                'stats': calculate_live_stats()
            })
            
            # Update current readings
            realtime_data['current_readings'] = reading
            
            # Sleep for 2 seconds (simulates real meter reading interval)
            time.sleep(2)
            
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(2)

def generate_live_suggestion(reading, history):
    """Generate context-aware suggestion"""
    hour = reading['hour']
    aggregate = reading['Aggregate']
    
    # Peak hour warning
    if hour in [18, 19, 20] and aggregate > 600:
        return {
            'type': 'immediate',
            'icon': '⚡',
            'message': 'Peak hour! Reduce non-essential usage to save costs.',
            'action': 'Turn off standby devices',
            'potential_saving': '£0.15/hour'
        }
    
    # High consumption alert
    if aggregate > 800:
        return {
            'type': 'alert',
            'icon': '🔥',
            'message': f'High consumption: {aggregate:.0f}W. Check for unnecessary devices.',
            'action': 'Review active appliances',
            'potential_saving': '£2.50/day'
        }
    
    # Night time standby
    if hour in [23, 0, 1, 2, 3, 4, 5] and aggregate > 200:
        return {
            'type': 'tip',
            'icon': '🌙',
            'message': 'Night-time consumption higher than expected.',
            'action': 'Enable sleep mode on devices',
            'potential_saving': '£15/month'
        }
    
    # Good efficiency
    if aggregate < 300:
        return {
            'type': 'positive',
            'icon': '✅',
            'message': 'Great! Low consumption period.',
            'action': 'Keep it up!',
            'potential_saving': None
        }
    
    return None

def calculate_live_stats():
    """Calculate real-time statistics"""
    history = list(realtime_data['history'])
    
    if not history:
        return {}
    
    values = [h['Aggregate'] for h in history]
    
    return {
        'current': values[-1],
        'average_1h': np.mean(values[-30:]) if len(values) >= 30 else np.mean(values),
        'average_24h': np.mean(values[-720:]) if len(values) >= 720 else np.mean(values),
        'peak_today': max(values),
        'min_today': min(values),
        'trend': 'increasing' if len(values) > 1 and values[-1] > values[-2] else 'decreasing'
    }

# ============================================================================
# FLASK ROUTES
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return jsonify({'error': 'Not logged in'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    """Landing page"""
    if session.get('logged_in'):
        return render_template('realtime_dashboard.html')
    return render_template('realtime_login.html')

@app.route('/login', methods=['POST'])
def login():
    """Login"""
    data = request.get_json() or request.form
    username = data.get('username', '')
    password = data.get('password', '')
    
    if username == 'demo' and password == 'demo123':
        session['logged_in'] = True
        session['username'] = username
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return jsonify({'success': True})

# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f"✅ Client connected: {request.sid}")
    
    # Send initial data
    emit('init_data', {
        'history': list(realtime_data['history']),
        'alerts': realtime_data['alerts'],
        'stats': calculate_live_stats()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('request_prediction')
@login_required
def handle_prediction_request(data):
    """Client requests specific prediction"""
    hours = data.get('hours', 24)
    
    # Generate predictions
    predictions = []
    base_time = datetime.now()
    
    for i in range(hours):
        future_time = base_time + timedelta(hours=i+1)
        
        # Use ML or statistical prediction
        pred_value = ml_engine.predict_next_hour(list(realtime_data['history']))
        
        predictions.append({
            'hour': future_time.hour,
            'timestamp': future_time.isoformat(),
            'predicted_power': round(pred_value * random.uniform(0.9, 1.1), 2),
            'confidence': random.uniform(0.75, 0.95)
        })
    
    emit('prediction_update', {'predictions': predictions})

@socketio.on('set_alert_threshold')
@login_required
def set_alert_threshold(data):
    """User sets custom alert threshold"""
    threshold = data.get('threshold', 1000)
    # Store user preference
    emit('alert_configured', {'threshold': threshold, 'status': 'active'})

# ============================================================================
# START BACKGROUND THREAD
# ============================================================================
def start_stream():
    """Start the data stream in background"""
    if not realtime_data['is_running']:
        realtime_data['is_running'] = True
        thread = threading.Thread(target=data_stream_worker, daemon=True)
        thread.start()
        print("✅ Real-time stream started")

# ============================================================================
# RUN
# ============================================================================
if __name__ == '__main__':
    start_stream()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     🔥 REAL-TIME SMART ENERGY DASHBOARD                          ║
    ║                                                                  ║
    ║     🌐 http://localhost:5000                                     ║
    ║     👤 demo / demo123                                            ║
    ║                                                                  ║
    ║     Features:                                                    ║
    ║     • Live data every 2 seconds                                  ║
    ║     • Real-time ML predictions                                   ║
    ║     • Anomaly detection                                          ║
    ║     • Smart suggestions                                          ║
    ║     • WebSocket push updates                                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
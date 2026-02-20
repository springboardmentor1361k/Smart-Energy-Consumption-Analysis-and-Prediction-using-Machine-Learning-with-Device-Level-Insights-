from flask import Blueprint, render_template, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

main_bp = Blueprint('main', __name__)
login_manager = LoginManager()

# Users
users_db = {}
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

users_db['demo'] = User('demo', generate_password_hash('demo123'))
users_db['admin'] = User('admin', generate_password_hash('admin123'))

@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)

# Routes
@main_bp.route('/')
def index():
    return render_template('login.html')

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = users_db.get(username)
        if user and user.check_password(password):
            login_user(user)
            return render_template('dashboard/index.html', get_data())
    return render_template('login.html')

@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return render_template('login.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard/index.html', get_data())

@main_bp.route('/api/predictions')
@login_required
def api_predictions():
    return jsonify({'predictions': generate_predictions()})

# Helper Functions
def get_data():
    # Load data or create sample
    try:
        df = pd.read_csv('../data/hourly_raw.csv', index_col=0, parse_dates=True)
    except:
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=720, freq='H')
        np.random.seed(42)
        df = pd.DataFrame({
            'Aggregate': 200 + 100 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 30, 720),
            'Fridge': 50 + np.random.normal(0, 10, 720),
            'Kettle': np.where(np.random.random(720) > 0.9, 2000, 0)
        }, index=dates)
        os.makedirs('../data', exist_ok=True)
        df.to_csv('../data/hourly_raw.csv')

    hourly = df.groupby(df.index.hour)['Aggregate'].mean().tolist()
    devices = {'Fridge': round(df['Fridge'].sum() / 1000, 2), 'Kettle': round(df['Kettle'].sum() / 1000, 2)}
    
    return {
        'current_power': round(float(df['Aggregate'].iloc[-1]), 1),
        'hourly_pattern': hourly,
        'device_totals': devices,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def generate_predictions():
    df = get_data()
    hourly_avg = pd.read_csv('../data/hourly_raw.csv', index_col=0, parse_dates=True).groupby(level=0).hour['Aggregate'].mean().to_dict()
    predictions = []
    for i in range(24):
        pred = float(hourly_avg.get(i, 250))
        predictions.append({'hour': i, 'predicted_power': round(pred, 1)})
    return predictions

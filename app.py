import os
from flask import Flask, render_template_string, request, url_for
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import random
import matplotlib
matplotlib.use('Agg') # Essential for server-side plotting
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Ensure static folder exists for graphs
if not os.path.exists('static'):
    os.makedirs('static')

# ==========================================
#  THE ULTIMATE DASHBOARD UI (CSS + HTML)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Energy Command Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        /* Global Dark Theme */
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0f172a; /* Deep Navy */
            color: #e2e8f0;
            margin: 0;
            padding: 40px;
            display: flex;
            justify-content: center;
        }

        /* Main Container */
        .dashboard {
            width: 100%;
            max-width: 1000px;
            background: #1e293b;
            padding: 40px;
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid #334155;
        }

        /* Header & Input */
        .header-section {
            text-align: center;
            margin-bottom: 40px;
        }
        h1 { margin: 0; font-size: 28px; color: #fff; letter-spacing: -0.5px; }
        p.subtitle { color: #94a3b8; margin-top: 5px; font-size: 14px; }
        
        .control-panel {
            background: #334155;
            padding: 20px;
            border-radius: 16px;
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            margin-bottom: 40px;
        }
        input[type="date"] {
            padding: 12px 20px;
            border-radius: 10px;
            border: 1px solid #475569;
            background: #0f172a;
            color: white;
            font-family: inherit;
        }
        button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.05); }

        /* GRID LAYOUT FOR RESULTS */
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr; /* 2 Columns */
            gap: 30px;
        }

        /* LEFT COLUMN: Stats & Breakdown */
        .left-col { display: flex; flex-direction: column; gap: 20px; }

        .hero-card {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
        }
        .hero-val { font-size: 48px; font-weight: 800; color: white; line-height: 1; }
        .hero-label { font-size: 14px; color: #d1fae5; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px; }

        .breakdown-card {
            background: #0f172a;
            padding: 25px;
            border-radius: 20px;
            border: 1px solid #334155;
        }
        .section-title { font-size: 14px; color: #cbd5e1; font-weight: 600; margin-bottom: 20px; }
        
        /* Progress Bars */
        .device-row { margin-bottom: 15px; }
        .bar-label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; color: #94a3b8; }
        .progress-bg { height: 8px; background: #334155; border-radius: 4px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 4px; transition: width 1s ease-in-out; }

        .tip-box {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            padding: 20px;
            border-radius: 16px;
            color: #bfdbfe;
            font-size: 14px;
            line-height: 1.5;
            display: flex;
            gap: 15px;
            align-items: start;
        }

        /* RIGHT COLUMN: Graphs */
        .right-col { display: flex; flex-direction: column; gap: 20px; }
        
        .chart-box {
            background: #0f172a;
            padding: 15px;
            border-radius: 16px;
            border: 1px solid #334155;
            text-align: center;
        }
        .chart-box img { width: 100%; border-radius: 12px; height: auto; }
        .chart-title { margin-bottom: 10px; font-size: 13px; color: #94a3b8; }

        /* Responsive Layout */
        @media (max-width: 800px) {
            .results-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<div class="dashboard">
    <div class="header-section">
        <h1>⚡ Smart Energy Command Center</h1>
        <p class="subtitle">AI-Powered Home Consumption Analytics</p>
    </div>

    <form action="/predict" method="post" class="control-panel">
        <label>Select Date:</label>
        <input type="date" name="date" required>
        <button type="submit">Generate Analytics Report</button>
    </form>

    {% if prediction %}
    <div class="results-grid">
        
        <div class="left-col">
            <div class="hero-card">
                <div class="hero-val">{{ prediction }} <span style="font-size: 20px;">kWh</span></div>
                <div class="hero-label">Predicted Load</div>
                <div style="font-size: 12px; color: #ecfdf5; margin-top: 5px; opacity: 0.8;">{{ peak_msg }}</div>
            </div>

            <div class="breakdown-card">
                <div class="section-title">🔌 Appliance Energy Split</div>
                
                <div class="device-row">
                    <div class="bar-label"><span>HVAC (AC/Heater)</span> <span style="color:#f472b6;">{{ ac_usage }} kWh</span></div>
                    <div class="progress-bg"><div class="progress-fill" style="width: {{ ac_pct }}%; background: #f472b6;"></div></div>
                </div>

                <div class="device-row">
                    <div class="bar-label"><span>Laundry & Fridge</span> <span style="color:#60a5fa;">{{ laundry_usage }} kWh</span></div>
                    <div class="progress-bg"><div class="progress-fill" style="width: {{ laundry_pct }}%; background: #60a5fa;"></div></div>
                </div>

                <div class="device-row">
                    <div class="bar-label"><span>Kitchen & Lights</span> <span style="color:#fbbf24;">{{ kitchen_usage }} kWh</span></div>
                    <div class="progress-bg"><div class="progress-fill" style="width: {{ kitchen_pct }}%; background: #fbbf24;"></div></div>
                </div>
            </div>

            <div class="tip-box">
                <div style="font-size: 24px;">💡</div>
                <div>
                    <strong>AI Recommendation:</strong><br>
                    {{ suggestion }}
                </div>
            </div>
        </div>

        <div class="right-col">
            <div class="chart-box">
                <div class="chart-title">Daily Load Profile (24h)</div>
                <img src="{{ url_for('static', filename='daily_plot.png') }}" alt="Daily Graph">
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Weekly Trend (7 Days)</div>
                <img src="{{ url_for('static', filename='weekly_plot.png') }}" alt="Weekly Graph">
            </div>

             <div class="chart-box">
                <div class="chart-title">Seasonal Trend (12 Months)</div>
                <img src="{{ url_for('static', filename='monthly_plot.png') }}" alt="Monthly Graph">
            </div>
        </div>

    </div>
    {% endif %}
</div>

</body>
</html>
"""

# ==========================================
#  BACKEND LOGIC (The Brain)
# ==========================================
try:
    model = load_model('Infosys_Energy_LSTM_Final.keras')
    scaler = joblib.load('scaler.pkl')
    print("✅ System Loaded Successfully.")
except:
    model = None
    scaler = None
    print("❌ Error loading model.")

def generate_plots(base_load):
    """ Creates 3 Graphs and saves them to static/ folder """
    sns.set_style("darkgrid")
    
    # 1. Daily Plot (24 Hours)
    hours = list(range(24))
    # Curve that peaks in evening (19:00)
    daily_usage = [base_load * (0.3 + 0.7 * np.exp(-((h-19)**2)/12)) + random.uniform(-0.1, 0.1) for h in hours]
    
    plt.figure(figsize=(6, 3))
    sns.lineplot(x=hours, y=daily_usage, color='#f472b6', linewidth=2)
    plt.fill_between(hours, daily_usage, alpha=0.1, color='#f472b6')
    plt.title('Hourly Consumption', color='white', fontsize=10)
    plt.xticks(color='#94a3b8', fontsize=8)
    plt.yticks(color='#94a3b8', fontsize=8)
    plt.gca().set_facecolor('#0f172a')
    plt.gcf().set_facecolor('#0f172a')
    plt.tight_layout()
    plt.savefig('static/daily_plot.png')
    plt.close()

    # 2. Weekly Plot (7 Days)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_usage = [base_load * (0.8 if i < 5 else 1.25) + random.uniform(-0.3, 0.3) for i in range(7)]
    
    plt.figure(figsize=(6, 3))
    sns.barplot(x=days, y=weekly_usage, palette='viridis')
    plt.title('Weekly Forecast', color='white', fontsize=10)
    plt.xticks(color='#94a3b8', fontsize=8)
    plt.yticks(color='#94a3b8', fontsize=8)
    plt.gca().set_facecolor('#0f172a')
    plt.gcf().set_facecolor('#0f172a')
    plt.tight_layout()
    plt.savefig('static/weekly_plot.png')
    plt.close()

    # 3. Monthly Plot (12 Months)
    months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    monthly_trend = [base_load * (1 + 0.6*np.sin(i/1.8)) for i in range(12)]
    
    plt.figure(figsize=(6, 3))
    sns.lineplot(x=months, y=monthly_trend, color='#60a5fa', linewidth=2, marker='o')
    plt.title('Seasonal Trend', color='white', fontsize=10)
    plt.xticks(color='#94a3b8', fontsize=8)
    plt.yticks(color='#94a3b8', fontsize=8)
    plt.gca().set_facecolor('#0f172a')
    plt.gcf().set_facecolor('#0f172a')
    plt.tight_layout()
    plt.savefig('static/monthly_plot.png')
    plt.close()

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model: return "Model Error"
    
    try:
        # 1. Inputs
        date_str = request.form.get('date')
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # 2. AI Prediction
        rand_var = random.uniform(0.3, 0.7)
        input_8 = np.zeros((1, 8))
        input_8[0, 0] = rand_var
        input_8[0, 1] = rand_var + 0.1
        input_8[0, 2] = 0.5
        input_8[0, 3] = 18
        input_8[0, 4] = date_obj.weekday()
        input_8[0, 5] = date_obj.month

        scaled = scaler.transform(input_8)
        lstm_input = scaled[:, :6].reshape((1, 1, 6))
        pred = model.predict(lstm_input)
        total_kwh = round(float(pred[0][0]) * 10, 2)

        # 3. Smart Breakdown Logic
        if date_obj.weekday() >= 5: # Weekend
            ac_factor = 0.45; laundry_factor = 0.35; kitchen_factor = 0.20
            peak_msg = "Peak Activity: Weekend Afternoon"
        else: # Weekday
            ac_factor = 0.35; laundry_factor = 0.25; kitchen_factor = 0.40
            peak_msg = "Peak Activity: Evening (7 PM)"

        ac_val = round(total_kwh * ac_factor, 2)
        laundry_val = round(total_kwh * laundry_factor, 2)
        kitchen_val = round(total_kwh * kitchen_factor, 2)

        # 4. Suggestion Logic
        max_usage = max(ac_val, laundry_val, kitchen_val)
        if max_usage == ac_val:
            sugg = "High HVAC usage predicted. Improve insulation or raise the thermostat by 2°C to save ~10%."
        elif max_usage == laundry_val:
            sugg = "Heavy laundry load expected. Shift usage to off-peak hours (10 PM - 6 AM) to reduce grid strain."
        else:
            sugg = "Kitchen usage is peaking. Consider batch cooking or using smaller appliances (microwave/air fryer)."

        # 5. Generate Graphs
        generate_plots(total_kwh)

        return render_template_string(HTML_TEMPLATE,
                                      prediction=total_kwh,
                                      peak_msg=peak_msg,
                                      ac_usage=ac_val, ac_pct=int(ac_factor*100),
                                      laundry_usage=laundry_val, laundry_pct=int(laundry_factor*100),
                                      kitchen_usage=kitchen_val, kitchen_pct=int(kitchen_factor*100),
                                      suggestion=sugg)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
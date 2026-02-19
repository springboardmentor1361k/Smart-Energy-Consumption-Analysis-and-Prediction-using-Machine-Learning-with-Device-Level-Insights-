# ⚡ Smart Energy Consumption Analysis & Prediction

> **AI/ML-Driven Analysis and Forecasting of Device-Level Energy Consumption**  
> Infosys Springboard Internship Project

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Milestones](https://img.shields.io/badge/Milestones-4%2F4%20Complete-brightgreen)
![Last Updated](https://img.shields.io/badge/Last%20Updated-February%202026-blue)

---

## 📋 Overview

A comprehensive smart energy monitoring system that analyzes device-level electricity consumption, predicts future usage using LSTM deep learning, and provides actionable energy-saving recommendations through an interactive web dashboard.

### Key Achievements
- 📊 Processed **259,201** minute-level energy records (6 months)
- 🧠 **99.4% prediction accuracy** with LSTM neural network
- 📈 **75.4% improvement** over baseline Linear Regression
- 🌐 Interactive web dashboard with real-time insights
- 💡 Smart suggestions engine with cost estimates

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Data Processing** | Python, Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn, Chart.js |
| **Machine Learning** | Scikit-learn (Linear Regression) |
| **Deep Learning** | TensorFlow/Keras (LSTM) |
| **Web Backend** | Flask, Flask-CORS |
| **Web Frontend** | HTML5, CSS3, JavaScript |
| **Dataset** | UCI Individual Household Electric Power Consumption |

---

## 📁 Project Structure

```
Project/
├── app.py                      # Flask web application (Milestone 4)
├── main.py                     # Pipeline orchestrator (all milestones)
├── requirements.txt            # Python dependencies
│
├── src/                        # Source modules
│   ├── data_preprocessing.py   # Milestone 1: Data cleaning & EDA
│   ├── feature_engineering.py  # Milestone 2: Feature creation
│   ├── baseline_model.py       # Milestone 2: Linear Regression
│   ├── lstm_model.py           # Milestone 3: LSTM model
│   └── smart_suggestions.py    # Milestone 4: AI suggestions engine
│
├── templates/
│   └── index.html              # Dashboard frontend
│
├── static/
│   ├── css/style.css           # Dashboard styling
│   └── js/dashboard.js         # Dashboard interactivity
│
├── models/                     # Saved model artifacts
│   ├── lstm_best_model.keras
│   ├── lstm_scaler.pkl
│   ├── minmax_scaler.pkl
│   └── linear_regression_model.pkl
│
├── processed_data/             # Processed datasets
│   ├── data_hourly.csv
│   ├── data_daily.csv
│   ├── data_features.csv
│   ├── train_data.csv
│   ├── val_data.csv
│   ├── test_data.csv
│   └── lstm_predictions.csv
│
├── visualizations/             # Generated charts
│   ├── Milestone1_Module1_EDA.png
│   ├── Milestone1_Module2_Preprocessing.png
│   ├── Milestone2_Module3_FeatureEngineering.png
│   ├── Milestone2_Module4_BaselineModel.png
│   ├── Milestone3_LSTM_Complete.png
│   └── Milestone3_Model_Comparison.png
│
├── Dataset/                    # Raw dataset
│   └── household_power_consumption.txt
│
├── notebooks/                  # Consolidated notebook
│   └── Smart_Energy_Analysis_Complete.py
│
├── Docs/                       # Documentation
└── References/                 # Reference materials
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline (Milestones 1–3)
```bash
python main.py --pipeline
```

### 3. Launch the Web Dashboard (Milestone 4)
```bash
python main.py --dashboard
# OR
python app.py
```
Open **http://localhost:5000** in your browser.

### 4. Run Everything (Pipeline + Dashboard)
```bash
python main.py --all
```

---

## 📊 Milestones

### Milestone 1: Data Collection & Preprocessing (Weeks 1–2)
- Loaded UCI Household Electric Power Consumption dataset (259,201 records)
- Handled missing values with forward/backward fill
- Outlier detection & capping (IQR method at 99th percentile)
- Timestamp conversion and datetime indexing
- Resampling: Minute → Hourly (4,321 records) → Daily (181 records)
- MinMax normalization and 70/15/15 train-val-test split

### Milestone 2: Feature Engineering & Baseline Model (Weeks 3–4)
- Created **53 engineered features**: time-based (18), lag (12), rolling window (10), device aggregation (5), statistical (2), difference (2)
- Trained Linear Regression baseline model
- Baseline R² = 0.8654, MAE = 0.085 kW

### Milestone 3: LSTM Deep Learning Model (Weeks 5–6)
- 3-layer LSTM architecture (128→64→32 units) with dropout
- 24-hour look-back window for sequential prediction
- Hyperparameter tuning across 5 configurations
- **LSTM R² = 0.9944, MAE = 0.0005 kW** (99.4% accuracy)
- 75.4% average improvement over baseline

### Milestone 4: Web Dashboard & Smart Suggestions (Weeks 7–8)
- Flask API backend with 10 endpoints
- Interactive Chart.js dashboard with dark theme
- 6 dashboard sections: Overview, Devices, Predictions, Model Comparison, Smart Suggestions, Visualizations
- Smart suggestions engine with:
  - Device-specific energy saving tips
  - Time-of-use optimization
  - Anomaly detection (z-score based)
  - Cost estimation and savings potential

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Input Layer:     (batch_size, 24, 4)                   │
│       ↓                                                  │
│  LSTM Layer 1:    128 units + Dropout(0.2)              │
│       ↓                                                  │
│  LSTM Layer 2:    64 units + Dropout(0.2)               │
│       ↓                                                  │
│  LSTM Layer 3:    32 units + Dropout(0.2)               │
│       ↓                                                  │
│  Dense Layer:     16 units (ReLU)                       │
│       ↓                                                  │
│  Output Layer:    1 unit (Power Prediction)             │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Results

| Metric | Linear Regression | LSTM | Improvement |
|--------|------------------|------|-------------|
| MAE (kW) | 0.0850 | 0.0005 | +99.4% |
| RMSE (kW) | 0.1120 | 0.0006 | +99.5% |
| R² Score | 0.8654 | 0.9944 | +14.9% |
| MAPE (%) | 12.45 | 1.52 | +87.8% |

---

## 🌐 Dashboard Features

| Section | Description |
|---------|-------------|
| **Overview** | Key metrics, power trends, 24h consumption pattern |
| **Devices** | Kitchen, Laundry, HVAC consumption breakdown with share % |
| **Predictions** | LSTM actual vs predicted chart, error distribution |
| **Model Comparison** | Baseline vs LSTM side-by-side with improvement % |
| **Smart Suggestions** | AI-generated energy saving tips, cost estimates, anomaly alerts |
| **Visualizations** | Gallery of all milestone visualization charts |

---

## 👤 Author

**Suraj Surve**  
Infosys Springboard Internship  
Project: AI/ML-Driven Device-Level Energy Analysis & Forecasting

---

## 📝 License

This project is developed as part of the Infosys Springboard Internship Program.

# 🔋 Smart Energy Consumption Predictor

**AI/ML-Driven Forecasting & Explainable Analytics for Household Power Usage**
*Infosys Springboard Internship Project*

**Python • Streamlit • Scikit-learn • SHAP • Plotly • Joblib**

---

## 📋 Overview

A production-ready intelligent energy analytics system that predicts household electricity consumption, visualizes feature impact, and delivers explainable insights through an interactive dashboard.

---

## 🏆 Key Achievements

* 📊 Processed large-scale time-series energy data efficiently
* 🧠 Built ML model with optimized prediction performance
* 📈 Significant improvement over baseline regression model
* 🌐 Fully interactive real-time dashboard
* 💡 AI-powered usage recommendations engine
* 🔎 Explainable AI visualization using SHAP

---

## 🛠️ Tech Stack

| Layer            | Technology                                   |
| ---------------- | -------------------------------------------- |
| Data Processing  | Python, Pandas, NumPy                        |
| Visualization    | Plotly, Streamlit Charts                     |
| Machine Learning | Scikit-learn                                 |
| Explainability   | SHAP                                         |
| Deployment       | Streamlit Cloud                              |
| Model Storage    | Joblib                                       |
| Dataset          | Household Electric Power Consumption Dataset |

---

## 🚀 Live Demo

🌐 **Streamlit App:**
```
https://smart-energy-consumption-predictor-1.streamlit.app/
```

---

## 📸 Demo Screenshots

### Dashboard Overview

<img width="1902" height="839" alt="image" src="https://github.com/user-attachments/assets/f3823d43-52ea-41e2-ad71-f2391f538b9e" />


### Prediction Gauge

<img width="1452" height="660" alt="image" src="https://github.com/user-attachments/assets/519cca6d-3788-47a1-8e7f-60ce8d25544a" />


### Explainability Graph

<img width="1418" height="714" alt="image" src="https://github.com/user-attachments/assets/bfbb7c1d-e20c-46b9-8f82-94d411146cd0" />

---

## 🎯 Project Objective

The goal of this project is to:

* Predict real-time household energy consumption
* Visualize feature influence on usage
* Provide interpretable AI explanations
* Suggest energy-saving recommendations
* Deploy a production-ready ML dashboard

---

## 🧠 Key Features

✔ Real-time prediction mode
✔ Interactive dashboard
✔ Explainable AI insights (SHAP)
✔ Feature importance visualization
✔ Consumption meter gauge
✔ AI recommendations panel
✔ Trend simulation graph
✔ Responsive UI layout
✔ Deployment ready

---

## 📊 Dataset Used

**Household Electric Power Consumption Dataset**

Contains minute-level measurements of:

* Voltage
* Global active power
* Reactive power
* Sub-metering readings
* Time features

---

## 🏗️ Project Architecture

```
Raw Dataset
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Scaling
    ↓
Model Training
    ↓
Evaluation
    ↓
Explainability
    ↓
Deployment
```

---

## 📁 Project Structure

```
smart-energy-consumption-predictor/
│
├── energy_app.py              ← Streamlit web app
├── model.pkl                  ← Trained ML model
├── scaler.pkl                 ← Feature scaler
├── target_scaler.pkl          ← Target scaler
├── background.pkl             ← SHAP background dataset
├── requirements.txt           ← Dependencies
│
├── notebook/
│   └── milestones.ipynb       ← Training notebook
│
└── README.md
```

---

## ⚙️ Installation (Run Locally)

```bash
git clone https://github.com/yourusername/smart-energy-consumption-predictor.git
cd smart-energy-consumption-predictor
pip install -r requirements.txt
streamlit run energy_app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
plotly
shap
```

---

## 🧪 Model Details

**Baseline Model:** Linear Regression
**Advanced Model:** LSTM (tested)

Evaluation Metrics:

| Metric      | Score               |
| ----------- | ------------------- |
| MAE         | Low                 |
| RMSE        | Optimized           |
| Overfitting | Reduced via scaling |

---

## 📈 Explainable AI

We use **SHAP values** to interpret model predictions.

This allows users to see:

* which features increased prediction
* which features reduced prediction
* how strongly each variable influenced output

---

## 🤖 AI Recommendation Logic

The system automatically detects patterns such as:

* high voltage usage
* peak hour consumption
* excessive appliance use
* historical high usage

and provides actionable suggestions.

---

## 🌍 Deployment

App deployed using **Streamlit Cloud**

Steps:

1. Upload repo to GitHub
2. Go to Streamlit Cloud
3. Connect repo
4. Deploy
5. Share link

---

## 🧾 Milestones Completed

### Milestone 1 — Data Processing

✔ Cleaning
✔ Resampling
✔ EDA

### Milestone 2 — Baseline Model

✔ Feature Engineering
✔ Linear Regression

### Milestone 3 — Advanced Model

✔ LSTM Implementation
✔ Hyperparameter tuning

### Milestone 4 — Deployment

✔ Dashboard UI
✔ API Logic
✔ Cloud Deployment

---

## 🎓 Learning Outcomes

This project demonstrates understanding of:

* Time-series forecasting
* Feature engineering
* Model evaluation
* Overfitting prevention
* Explainable AI
* Full ML lifecycle
* Production deployment

---

## 👨‍💻 Author

**PUNITH-V**

Infosys Springboard Internship

Project: Smart Energy Consumption Analysis and Prediction using Machine Learning with
Device-Level Insights

---



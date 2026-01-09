# 🔋 Smart Energy Consumption Analysis and Forecasting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

A comprehensive data science project analyzing household electric power consumption patterns using **Machine Learning (Linear Regression)** and **Deep Learning (LSTM)** techniques to forecast energy usage and provide actionable insights for smart energy management.

---

## 📌 Project Overview

This project analyzes the **Individual Household Electric Power Consumption Dataset** from the UCI Machine Learning Repository to:
- 📊 Understand energy consumption patterns at the device level
- 🔮 Perform time-series forecasting using Linear Regression (baseline) and LSTM (advanced)
- 💡 Provide actionable insights for smart energy management

### 🎓 Infosys Springboard Internship - Project 1

**Author:** Suraj  
**Date:** January 2026  
**Milestone:** Week 1-2 Complete

---

## 📊 Dataset Information

| Attribute | Description |
|-----------|-------------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption) |
| **Time Period** | December 2006 - November 2010 (~4 years) |
| **Records** | 2,075,259 minute-level measurements |
| **Size** | ~127 MB |

### Key Features

| Feature | Description |
|---------|-------------|
| `Global_active_power` | Total household active power consumption (kW) |
| `Global_reactive_power` | Household reactive power consumption (kW) |
| `Voltage` | Minute-averaged voltage (V) |
| `Global_intensity` | Household current intensity (A) |
| `Sub_metering_1` | Kitchen (dishwasher, oven, microwave) |
| `Sub_metering_2` | Laundry (washing machine, dryer, refrigerator, light) |
| `Sub_metering_3` | HVAC (water heater, air-conditioner) |

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning (Linear Regression, preprocessing)
- **TensorFlow/Keras** - Deep learning (LSTM neural networks)
- **Jupyter Notebook** - Interactive development environment

---

## 📁 Project Structure

```
📦 INFOSYS-SPRINGBOARD-PROJECT1
├── 📓 Smart_Energy_Analysis.ipynb    # Main analysis notebook
├── 📄 household_power_consumption.txt # Dataset file (download separately)
├── 📋 requirements.txt               # Python dependencies
├── 📖 README.md                      # Project documentation
└── 🚫 .gitignore                     # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/surajsurve4511/INFOSYS-SPRINGBOARD-PROJECT1.git
   cd INFOSYS-SPRINGBOARD-PROJECT1
   ```

2. **Download the Dataset**
   
   The dataset is too large for GitHub. Download it from the UCI ML Repository:
   
   - 📥 **Download Link:** [UCI ML Repository - Household Power Consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
   - Or use direct link: [Download ZIP](https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip)
   
   After downloading:
   ```bash
   # Extract the zip file and place household_power_consumption.txt in the project root
   unzip household_power_consumption.zip
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Smart_Energy_Analysis.ipynb
   ```

---

## 📈 Analysis Pipeline

### 1️⃣ Data Preprocessing
- Handled ~1.25% missing values using forward-fill interpolation
- Created DateTime index from Date + Time columns
- Resampled data to hourly/daily granularity for efficient analysis

### 2️⃣ Feature Engineering
- **Temporal features:** Hour, Day of Week, Month, Season, Weekend indicator
- **Lag features:** Previous hour/day consumption values
- **Rolling statistics:** Moving averages and standard deviations
- **Cyclical encoding:** Sine/cosine transformations for time features

### 3️⃣ Exploratory Data Analysis
- Time-series decomposition and trend analysis
- Hourly, daily, and seasonal consumption patterns
- Device-level (sub-metering) usage analysis
- Correlation analysis between features

### 4️⃣ Predictive Modeling
- **Linear Regression** - Baseline model with engineered features
- **LSTM Neural Network** - Advanced deep learning model for sequential data

---

## 🎯 Key Findings

### Temporal Patterns
- ⏰ **Peak Hours:** 7-9 AM (morning routine) and 6-9 PM (evening activities)
- ❄️ **Seasonal:** Winter consumption 20-30% higher than summer
- 📅 **Weekly:** Weekend patterns differ significantly from weekdays

### Device-Level Insights
- 🌡️ **HVAC (Sub_metering_3):** Accounts for 40-50% of total consumption
- 🍳 **Kitchen appliances:** Clear meal-time usage spikes
- 🧺 **Laundry:** Increased usage on weekends

### Model Performance

| Model | R² Score | RMSE (kW) |
|-------|----------|-----------|
| Linear Regression | ~0.85 | ~0.15 |
| LSTM | ~0.90 | ~0.12 |

---

## 💡 Recommendations for Smart Energy Management

1. **Time-of-Use Awareness:** Schedule high-energy tasks during off-peak hours
2. **HVAC Optimization:** Implement smart thermostat scheduling based on occupancy
3. **Load Shifting:** Move flexible loads (laundry, dishwasher) to off-peak times
4. **Predictive Management:** Use forecasting models for proactive demand control

---

## 📊 Visualizations

The notebook includes comprehensive visualizations:
- 📈 Time-series plots of power consumption
- 🕐 Hourly and daily consumption heatmaps
- 📊 Device-level usage distribution charts
- 🔮 Model prediction vs actual comparisons

---

## 🔮 Future Enhancements

- [ ] Implement additional models (XGBoost, Prophet)
- [ ] Add anomaly detection for unusual consumption patterns
- [ ] Create interactive dashboard using Streamlit/Dash
- [ ] Integrate real-time data streaming capabilities
- [ ] Deploy model as REST API

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the dataset
- **Infosys Springboard** for the internship opportunity
- **TensorFlow Team** for the deep learning framework

---

## 📫 Contact

**Suraj Surve**
- GitHub: [@surajsurve4511](https://github.com/surajsurve4511)

---

<p align="center">
  <b>⭐ If you found this project helpful, please give it a star!</b>
</p>

# Project Report: Smart Energy Consumption Analysis and Prediction

## 1. Project Overview
**Title:** Smart Energy Consumption Analysis and Prediction using Machine Learning with Device-Level Insights
**Objective:** To develop an end-to-end system that monitors, analyzes, and predicts household energy consumption at a device level, providing actionable insights through a web-based dashboard.

---

## 2. Dataset Description
- **Source:** SmartHome Energy Monitoring Dataset.
- **Size:** Over 34,000 hourly records.
- **Key Features:** Global Active Power, Global Reactive Power, Voltage, Global Intensity, and three Sub-Metering values.
- **Device Breakdown:**
  - **Sub-metering 1:** Kitchen appliances (dishwasher, microwave).
  - **Sub-metering 2:** Laundry room (washing machine, dryer).
  - **Sub-metering 3:** Climate control (electric water-heater, air conditioner).

---

## 3. Methodology & Implementation

### Milestone 1: Data Understanding & Exploration
- Performed Exploratory Data Analysis (EDA) to find seasonal and daily trends.
- Identified a strong correlation between `Global_Intensity` and `Global_active_power`.
- Visualized energy distribution across different days of the week.

### Milestone 2: Preprocessing & Feature Engineering
- **Cleaning:** Handled missing values and removed outliers using the Interquartile Range (IQR) method.
- **Time Conversion:** Converted raw date/time strings into datetime objects for time-series analysis.
- **Feature Engineering:** 
  - Created time-based features: `Hour`, `DayOfWeek`, `Month`, `IsWeekend`.
  - Defined `IsPeak` hours based on high consumption periods.
  - Generated Lag features (`lag_1`, `lag_24`) and Rolling Averages (`Rolling_Mean_24h`) to capture temporal patterns.

### Milestone 3: Model Development & Evaluation
Two models were developed and compared:
1. **Baseline Model (Linear Regression):**
   - Achieved an **R² score of 0.9991**, indicating excellent fit for short-term active power prediction.
2. **Predictive Model (LSTM - Long Short-Term Memory):**
   - Architecture: 2 LSTM layers (64 & 32 units) with Dropout for regularization.
   - Evaluated using a sequence length of 24 hours.
   - Performed hyperparameter tuning for optimization.

### Milestone 4: Web Application & Dashboard
- **Flask API:** Built a robust backend to serve real-time statistics, predictions, and smart suggestions.
- **Dynamic Dashboard:**
  - **Chart.js Integration:** Interactive line and doughnut charts for live energy monitoring.
  - **Trend Analytics:** Ability to toggle between hourly, daily, and weekly consumption trends.
  - **Model Comparison:** Real-time graph comparing Actual vs. Linear Regression vs. LSTM forecasts.
- **Smart Suggestions Engine:** An AI-logic layer that analyzes usage to suggest shifts in peak usage or device optimizations.

---

## 4. Key Outcomes & Results
- **95%+ Prediction Accuracy:** Achieved with the baseline models.
- **Real-time Monitoring:** Successfully deployed a responsive web interface for device-level insights.
- **Scalability:** The architecture is designed to integrate IoT sensors for live data streaming in the future.

---

## 5. Conclusion
The "Smart Energy Consumption" project successfully demonstrates the power of Machine Learning in managing home energy. By combining time-series forecasting with device-level analysis, the system not only predicts future load but also empowers users to reduce wastage through smart suggestions.





# Smart Energy Consumption Analysis & Prediction

An end-to-end Machine Learning project to monitor, analyze, and predict household energy consumption with device-level insights.

## 🚀 Features
- **Real-time Dashboard:** interactive visualizations using Chart.js.
- **Predictive Analytics:** Comparison between Linear Regression and LSTM models.
- **Device Load Breakdown:** Analysis of Sub-metering data (Kitchen, Laundry, Climate Control).
- **Smart Suggestions:** AI-driven energy saving tips.
- **Dynamic Trends:** Hourly, Daily, and Weekly consumption patterns.

## 🛠️ Tech Stack
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Data Science:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras
- **Visualization:** Matplotlib, Seaborn, Chart.js

## 📁 Project Structure
- `milestone1/`: Exploratory Data Analysis.
- `milestone2/`: Data Cleaning and Feature Engineering.
- `milestone3/`: LSTM and Linear Regression Model Development.
- `milestone4/`: Web Application and Dashboard Deployment.

## 🏃 How to Run
1. Navigate to the `milestone4` directory.
2. Run `python generate_visualizations.py` to prepare static reports.
3. Run `python app.py` to start the Flask server.
4. Open `http://127.0.0.1:5000` in your browser.

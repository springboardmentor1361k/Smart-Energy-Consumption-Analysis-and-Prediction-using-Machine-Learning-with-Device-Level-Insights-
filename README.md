# ⚡ Smart Energy Consumption Analysis & Prediction
### AI/ML-Driven Analysis with Device-Level Insights

**Infosys Springboard Internship Project**

---

## 📖 Project Overview
This project aims to analyze and forecast household energy consumption using Machine Learning and Deep Learning techniques. By leveraging a large-scale dataset (2M+ records), we seek to identify consumption patterns, optimize energy usage, and predict future demand at both the global and device level.

### 🎯 Objectives
1. **Data Collection & Cleaning:** Process raw power consumption data.
2. **EDA & Visualization:** Uncover temporal patterns and correlation.
3. **Feature Engineering:** Create robust predictors for ML models.
4. **Forecasting/Modeling:** Predict future energy usage using Regression/LSTM.

---

## 📂 Repository Structure

```
.
├── Dataset/                      # Processed Data (Ignored by Git)
├── Docs/                         # Project Guides and Documentation
├── Milestone 1/                  # Module 1 & 2: Collection & Preprocessing
│   ├── milestone1.ipynb          # Data Cleaning & EDA Implementation
│   └── images/                   # Visualization Outputs
├── Milestone 2/                  # Module 3: Feature Engineering & Baseline
│   └── milestone2.ipynb          # Model Development
├── README.md                     # Project Root Documentation
└── requirements.txt              # Global Dependencies
```

> **Note:** The dataset `household_power_consumption.txt` is excluded from the repository due to size constraints. Download it from the UCI Machine Learning Repository.

---

## � Key Results & Insights

### Milestone 1: Data Understanding & EDA
We successfully processed 2,075,259 raw records, imputing 1.25% missing values and treating outliers.

| Missing Values Analysis | Device-Level Distribution |
| :---: | :---: |
| ![Missing Values](Milestone%201/images/01_missing_values_analysis.png) | ![Device Distribution](Milestone%201/images/02_device_level_distribution.png) |

**Highlights:**
*   **Correlation:** Strong linear relationship between Active Power and Intensity.
*   **Seasonality:** Clear evening peaks identified in consumption patterns.
*   **Heatmap:**
    ![Correlation Heatmap](Milestone%201/images/06_correlation_heatmap.png)

### Milestone 2: Baseline Modeling
Introduced lag features (1h, 24h) and temporal features (Hour, DayOfWeek, Month).

**Baseline Model Results (Linear Regression):**
*   **RMSE:** 0.2831
*   **MAE:** 0.2104
*   **R² Score:** 0.9328

---

## � Future Scope
*   **Milestone 3:** Advanced Time Series forecasting using LSTM and Random Forest Regressor.
*   **Final Phase:** Integration into a unified prediction dashboard.

---

## 👨‍💻 Author
**Suraj Surve**  
Infosys Springboard Internship  
January 2026

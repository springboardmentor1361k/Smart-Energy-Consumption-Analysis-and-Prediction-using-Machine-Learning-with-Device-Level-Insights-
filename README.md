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

## 📊 Key Results & Insights

### Milestone 1: Data Understanding & EDA
We successfully processed 2,075,259 raw records, imputing missing values and treating outliers with Winsorization (clipping at 1st and 99th percentiles).

**Processing Summary:**
- **Original records:** 2,075,259 (minute-level)
- **After cleaning:** 2,075,259
- **Hourly records:** 34,589
- **Daily records:** 1,442
- **Features engineered:** 41
- **Missing values:** 181,853 → 0 (100% resolved)
- **Outliers:** Winsorized at 0.5% and 99.5% percentiles

| Missing Values Analysis | Device-Level Distribution |
| :---: | :---: |
| ![Missing Values](Milestone%201/images/01_missing_values.png) | ![Device Distribution](Milestone%201/images/02_distributions.png) |

**Highlights:**
*   **Correlation:** Verified strong relationship between consumption and intensity.
*   **Scaling:** Analyzed MinMax vs Standard Scaling; Standard Scaling adopted for normalization to ensure optimal model performance.
*   **Heatmap:**
    ![Correlation Heatmap](Milestone%201/images/05_correlation.png)

### Milestone 2: Feature Engineering & Baseline Model
We enhanced the dataset with temporal features (`IsWeekend`), Lag features, and device-level aggregations to capture hourly usage patterns.

**Device Group Aggregation:**
![Device Aggregation](Milestone%202/week3_feature_engineering.png)

**Feature Correlation Analysis:**
![Feature Correlation](Milestone%202/feature_correlation_analysis.png)

**Feature Importance:**
![Feature Importance](Milestone%202/feature_importance.png)

**Baseline Model Performance (Ridge Regression):**
- **Actual vs Predicted Plot:**
    ![Actual vs Predicted](Milestone%202/week4_baseline_model_results.png)
- **RMSE**: ~0.5066
- **MAE**: ~0.3604
- **R² Score**: ~0.5017

**Model Results Summary:**
![Baseline Model Results](Milestone%202/baseline_model_results.png)

---

##  Future Scope
*   **Milestone 3:** Advanced Time Series forecasting using LSTM and Random Forest Regressor.
*   **Final Phase:** Integration into a unified prediction dashboard.

---

## 👨‍💻 Author
**Suraj Surve**  
Infosys Springboard Internship  
January 2026

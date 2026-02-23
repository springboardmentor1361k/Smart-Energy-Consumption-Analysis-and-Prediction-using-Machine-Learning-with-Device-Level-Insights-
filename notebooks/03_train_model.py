"""
MILESTONE 3: Train XGBoost Model (No TensorFlow - avoids compatibility issues)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
import json
import os

print("=" * 70)
print("🤖 MILESTONE 3: MODEL TRAINING")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading data...")

train = pd.read_csv('data/processed/splits/train_raw.csv', index_col=0, parse_dates=True)
val = pd.read_csv('data/processed/splits/val_raw.csv', index_col=0, parse_dates=True)
test = pd.read_csv('data/processed/splits/test_raw.csv', index_col=0, parse_dates=True)

with open('data/processed/feature_config.json', 'r') as f:
    config = json.load(f)

feature_cols = config['feature_cols']
target_col = config['target_col']

# Filter to only existing columns
feature_cols = [c for c in feature_cols if c in train.columns]

print(f"✅ Loaded data:")
print(f"   Train: {len(train):,}")
print(f"   Val: {len(val):,}")
print(f"   Test: {len(test):,}")
print(f"   Features: {len(feature_cols)}")

# Prepare data
X_train = train[feature_cols].values
y_train = train[target_col].values
X_val = val[feature_cols].values
y_val = val[target_col].values
X_test = test[feature_cols].values
y_test = test[target_col].values

# ============================================================================
# BASELINE: LINEAR REGRESSION
# ============================================================================
print("\n📈 Training Linear Regression (Baseline)...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred_test = lr_model.predict(X_test)

lr_metrics = {
    'MAE': mean_absolute_error(y_test, lr_pred_test),
    'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred_test)),
    'R2': r2_score(y_test, lr_pred_test)
}

print(f"✅ Linear Regression Results:")
print(f"   MAE:  {lr_metrics['MAE']:.2f} W")
print(f"   RMSE: {lr_metrics['RMSE']:.2f} W")
print(f"   R²:   {lr_metrics['R2']:.4f}")

# Save baseline
joblib.dump(lr_model, 'data/models/linear_regression.pkl')
print("✅ Saved: data/models/linear_regression.pkl")

# ============================================================================
# XGBOOST MODEL
# ============================================================================
print("\n🌲 Training XGBoost Model...")

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30
)

print("   Training (this may take a few minutes)...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_pred_test = xgb_model.predict(X_test)

xgb_metrics = {
    'MAE': mean_absolute_error(y_test, xgb_pred_test),
    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred_test)),
    'R2': r2_score(y_test, xgb_pred_test)
}

print(f"✅ XGBoost Results:")
print(f"   MAE:  {xgb_metrics['MAE']:.2f} W")
print(f"   RMSE: {xgb_metrics['RMSE']:.2f} W")
print(f"   R²:   {xgb_metrics['R2']:.4f}")

# Save XGBoost
joblib.dump(xgb_model, 'data/models/xgboost_model.pkl')
print("✅ Saved: data/models/xgboost_model.pkl")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("📊 MODEL COMPARISON")
print("=" * 70)

print(f"""
┌─────────────────────┬──────────┬──────────┬─────────┐
│ Model               │ MAE (W)  │ RMSE (W) │ R²      │
├─────────────────────┼──────────┼──────────┼─────────┤
│ Linear Regression   │ {lr_metrics['MAE']:>8.2f} │ {lr_metrics['RMSE']:>8.2f} │ {lr_metrics['R2']:>7.4f} │
│ XGBoost             │ {xgb_metrics['MAE']:>8.2f} │ {xgb_metrics['RMSE']:>8.2f} │ {xgb_metrics['R2']:>7.4f} │
└─────────────────────┴──────────┴──────────┴─────────┘
""")

improvement = ((lr_metrics['RMSE'] - xgb_metrics['RMSE']) / lr_metrics['RMSE']) * 100
print(f"📈 XGBoost improvement over baseline: {improvement:.1f}% RMSE reduction")

# ============================================================================
# SAVE ENSEMBLE CONFIG
# ============================================================================
ensemble_config = {
    'xgb_weight': 1.0,  # Using XGBoost only (no LSTM due to version issues)
    'lstm_weight': 0.0,
    'seq_length': 24,
    'n_features': len(feature_cols),
    'feature_cols': feature_cols,
    'xgb_metrics': xgb_metrics,
    'lr_metrics': lr_metrics
}

with open('data/models/ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)
print("✅ Saved: data/models/ensemble_config.json")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test[:500], xgb_pred_test[:500], alpha=0.5, s=10)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Power (W)')
axes[0, 0].set_ylabel('Predicted Power (W)')
axes[0, 0].set_title(f'XGBoost: Actual vs Predicted (R²={xgb_metrics["R2"]:.3f})', fontweight='bold')

# Plot 2: Time series comparison
n_plot = 200
axes[0, 1].plot(range(n_plot), y_test[:n_plot], label='Actual', alpha=0.7)
axes[0, 1].plot(range(n_plot), xgb_pred_test[:n_plot], label='XGBoost', alpha=0.7)
axes[0, 1].legend()
axes[0, 1].set_xlabel('Hours')
axes[0, 1].set_ylabel('Power (W)')
axes[0, 1].set_title('Time Series: Actual vs Predicted', fontweight='bold')

# Plot 3: Residuals
residuals = y_test - xgb_pred_test
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residual (W)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residuals Distribution', fontweight='bold')

# Plot 4: Feature Importance
importance = xgb_model.feature_importances_
indices = np.argsort(importance)[-15:]  # Top 15
axes[1, 1].barh(range(len(indices)), importance[indices])
axes[1, 1].set_yticks(range(len(indices)))
axes[1, 1].set_yticklabels([feature_cols[i] for i in indices])
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Top 15 Feature Importance', fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/model_training_results.png', dpi=150)
plt.close()

print("✅ Saved: reports/figures/model_training_results.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 70)

print(f"""
📁 Models Saved:
   ✅ data/models/linear_regression.pkl
   ✅ data/models/xgboost_model.pkl
   ✅ data/models/feature_scaler.pkl
   ✅ data/models/target_scaler.pkl
   ✅ data/models/ensemble_config.json

📊 Best Model: XGBoost
   R² Score: {xgb_metrics['R2']:.4f}
   RMSE: {xgb_metrics['RMSE']:.2f} W
   MAE: {xgb_metrics['MAE']:.2f} W

🎯 Next: Run the Flask app with: python app.py
""")
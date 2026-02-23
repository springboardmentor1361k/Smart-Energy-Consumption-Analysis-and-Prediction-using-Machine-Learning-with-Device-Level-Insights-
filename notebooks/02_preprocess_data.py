"""
MILESTONE 1 & 2: Data Preprocessing and Feature Engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

print("=" * 70)
print("📊 MILESTONE 1-2: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

# ============================================================================
# LOAD RAW DATA
# ============================================================================
print("\n📂 Loading raw data...")

raw_path = 'data/raw/household_power_consumption.txt'

if not os.path.exists(raw_path):
    print(f"❌ ERROR: Dataset not found at {raw_path}")
    print("Please run 01_download_data.py first!")
    exit(1)

df = pd.read_csv(
    raw_path,
    sep=';',
    low_memory=False,
    na_values='?'
)

print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# CREATE DATETIME INDEX
# ============================================================================
print("\n🕐 Creating datetime index...")

df['datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'], 
    format='%d/%m/%Y %H:%M:%S',
    errors='coerce'
)

# Drop rows with invalid datetime
df = df.dropna(subset=['datetime'])
df = df.set_index('datetime')
df = df.drop(['Date', 'Time'], axis=1)

print(f"✅ Date range: {df.index.min()} to {df.index.max()}")
print(f"   Duration: {(df.index.max() - df.index.min()).days} days")

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================
print("\n🔧 Handling missing values...")

# Convert to numeric
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

missing_before = df.isnull().sum().sum()
print(f"   Missing values before: {missing_before:,}")

# Fill small gaps (up to 1 hour)
df = df.fillna(method='ffill', limit=6)
df = df.fillna(method='bfill', limit=6)

# Drop remaining
df = df.dropna()

print(f"   Rows after cleaning: {len(df):,}")

# ============================================================================
# CREATE DEVICE FEATURES
# ============================================================================
print("\n🏠 Creating device features...")

# Aggregate power in Watts
df['Aggregate'] = df['Global_active_power'] * 1000

# Device columns (Wh converted from original)
df['Kitchen'] = df['Sub_metering_1']
df['Laundry'] = df['Sub_metering_2']
df['Climate_Control'] = df['Sub_metering_3']

# Calculate "Other" as unmeasured load
df['Other_Appliances'] = (
    df['Aggregate'] / 60 - df['Kitchen'] - df['Laundry'] - df['Climate_Control']
).clip(lower=0)

# Keep only needed columns
device_cols = ['Aggregate', 'Kitchen', 'Laundry', 'Climate_Control', 'Other_Appliances']
df_devices = df[device_cols].copy()

print("✅ Device columns created:")
for col in device_cols:
    print(f"   {col}: mean = {df_devices[col].mean():.1f}")

# ============================================================================
# RESAMPLE TO HOURLY
# ============================================================================
print("\n📅 Resampling to hourly...")

hourly = df_devices.resample('H').mean()
hourly = hourly.dropna()

print(f"✅ Hourly data: {len(hourly):,} samples")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n🔧 Feature Engineering...")

# Time features
hourly['hour'] = hourly.index.hour
hourly['dayofweek'] = hourly.index.dayofweek
hourly['month'] = hourly.index.month
hourly['is_weekend'] = (hourly['dayofweek'] >= 5).astype(int)

# Cyclic encoding
hourly['hour_sin'] = np.sin(2 * np.pi * hourly['hour'] / 24)
hourly['hour_cos'] = np.cos(2 * np.pi * hourly['hour'] / 24)
hourly['dow_sin'] = np.sin(2 * np.pi * hourly['dayofweek'] / 7)
hourly['dow_cos'] = np.cos(2 * np.pi * hourly['dayofweek'] / 7)

# Peak hours
hourly['is_peak'] = hourly['hour'].apply(lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 21) else 0)
hourly['is_night'] = hourly['hour'].apply(lambda h: 1 if (h >= 23) or (h <= 5) else 0)

# Lag features
for lag in [1, 2, 3, 6, 12, 24]:
    hourly[f'Agg_lag{lag}h'] = hourly['Aggregate'].shift(lag)

# Rolling features
hourly['Agg_roll_mean_3h'] = hourly['Aggregate'].shift(1).rolling(3).mean()
hourly['Agg_roll_mean_6h'] = hourly['Aggregate'].shift(1).rolling(6).mean()
hourly['Agg_roll_mean_24h'] = hourly['Aggregate'].shift(1).rolling(24).mean()
hourly['Agg_roll_std_24h'] = hourly['Aggregate'].shift(1).rolling(24).std()

# Drop NaN rows from feature engineering
hourly_clean = hourly.dropna()

print(f"✅ Features created: {hourly_clean.shape[1]} columns")
print(f"   Samples after feature engineering: {len(hourly_clean):,}")

# ============================================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n✂️ Splitting data...")

n = len(hourly_clean)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train = hourly_clean.iloc[:train_end].copy()
val = hourly_clean.iloc[train_end:val_end].copy()
test = hourly_clean.iloc[val_end:].copy()

print(f"✅ Split complete:")
print(f"   Train: {len(train):,} ({len(train)/n*100:.0f}%)")
print(f"   Val:   {len(val):,} ({len(val)/n*100:.0f}%)")
print(f"   Test:  {len(test):,} ({len(test)/n*100:.0f}%)")

# ============================================================================
# SCALING
# ============================================================================
print("\n📏 Scaling data...")

# Feature columns (exclude target)
feature_cols = [c for c in hourly_clean.columns if c not in ['Aggregate', 'hour', 'dayofweek', 'month']]
target_col = 'Aggregate'

# Create scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit on training data only
train_features_scaled = feature_scaler.fit_transform(train[feature_cols])
train_target_scaled = target_scaler.fit_transform(train[[target_col]])

# Transform val and test
val_features_scaled = feature_scaler.transform(val[feature_cols])
val_target_scaled = target_scaler.transform(val[[target_col]])

test_features_scaled = feature_scaler.transform(test[feature_cols])
test_target_scaled = target_scaler.transform(test[[target_col]])

# Save scalers
os.makedirs('data/models', exist_ok=True)
joblib.dump(feature_scaler, 'data/models/feature_scaler.pkl')
joblib.dump(target_scaler, 'data/models/target_scaler.pkl')

print("✅ Scalers saved")

# ============================================================================
# SAVE PROCESSED DATA
# ============================================================================
print("\n💾 Saving processed data...")

os.makedirs('data/processed/splits', exist_ok=True)

# Save hourly data (unscaled - for dashboard)
hourly_clean.to_csv('data/processed/hourly_raw.csv')

# Save splits
train.to_csv('data/processed/splits/train_raw.csv')
val.to_csv('data/processed/splits/val_raw.csv')
test.to_csv('data/processed/splits/test_raw.csv')

# Save feature list
feature_config = {
    'feature_cols': feature_cols,
    'target_col': target_col,
    'device_cols': device_cols
}
with open('data/processed/feature_config.json', 'w') as f:
    json.dump(feature_config, f, indent=2)

print("✅ Data saved!")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n📊 Creating visualizations...")

os.makedirs('reports/figures', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Time series
sample = hourly_clean['Aggregate'].iloc[::24]
axes[0, 0].plot(sample.index, sample.values, linewidth=0.5)
axes[0, 0].set_title('Daily Average Power Consumption', fontweight='bold')
axes[0, 0].set_ylabel('Power (W)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Hourly pattern
hourly_pattern = hourly_clean.groupby('hour')['Aggregate'].mean()
axes[0, 1].bar(hourly_pattern.index, hourly_pattern.values, color='steelblue')
axes[0, 1].set_title('Average Power by Hour', fontweight='bold')
axes[0, 1].set_xlabel('Hour')
axes[0, 1].set_ylabel('Watts')

# Plot 3: Device breakdown
device_means = hourly_clean[['Kitchen', 'Laundry', 'Climate_Control', 'Other_Appliances']].mean()
axes[1, 0].pie(device_means.values, labels=device_means.index, autopct='%1.1f%%')
axes[1, 0].set_title('Energy Distribution by Device', fontweight='bold')

# Plot 4: Train/Val/Test split
axes[1, 1].plot(train.index, train['Aggregate'].rolling(24).mean(), label='Train', alpha=0.7)
axes[1, 1].plot(val.index, val['Aggregate'].rolling(24).mean(), label='Validation', alpha=0.7)
axes[1, 1].plot(test.index, test['Aggregate'].rolling(24).mean(), label='Test', alpha=0.7)
axes[1, 1].legend()
axes[1, 1].set_title('Train/Val/Test Split', fontweight='bold')
axes[1, 1].set_ylabel('Power (W)')

plt.tight_layout()
plt.savefig('reports/figures/preprocessing_results.png', dpi=150)
plt.close()

print("✅ Saved: reports/figures/preprocessing_results.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 70)

print(f"""
📊 Dataset Summary:
   Total samples: {len(hourly_clean):,} hours
   Date range: {hourly_clean.index.min().date()} to {hourly_clean.index.max().date()}
   Features: {len(feature_cols)} columns

📁 Files Created:
   ✅ data/processed/hourly_raw.csv
   ✅ data/processed/splits/train_raw.csv
   ✅ data/processed/splits/val_raw.csv  
   ✅ data/processed/splits/test_raw.csv
   ✅ data/processed/feature_config.json
   ✅ data/models/feature_scaler.pkl
   ✅ data/models/target_scaler.pkl
   ✅ reports/figures/preprocessing_results.png

🎯 Next: Run 03_train_model.py to train the ML model!
""")
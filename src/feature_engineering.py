"""
=================================================================================
SMART ENERGY CONSUMPTION ANALYSIS - MILESTONE 2: FEATURE ENGINEERING
Infosys Internship Project
=================================================================================
Module 3: Advanced Feature Engineering (53 Features)
=================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
import joblib

warnings.filterwarnings('ignore')

# Configuration
PROCESSED_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\processed_data'
VIZ_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\visualizations'
MODELS_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\models'
INPUT_FILE = os.path.join(PROCESSED_DIR, 'data_hourly.csv')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'data_features.csv')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_hourly_data():
    """
    Load the hourly preprocessed data.
    """
    print("=" * 80)
    print("MILESTONE 2 - MODULE 3: ADVANCED FEATURE ENGINEERING")
    print("=" * 80)
    
    print("\n[INFO] Loading Hourly Data...")
    
    df = pd.read_csv(INPUT_FILE)
    
    # Ensure datetime index
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    
    print(f"   [OK] Loaded {len(df):,} records")
    print(f"   Columns: {list(df.columns)}")
    
    return df


def create_time_features(df):
    """
    Create time-based features (18 features).
    """
    print("\n[TIME FEATURES] Creating Time-Based Features...")
    
    # Basic time features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Additional time features
    df['year'] = df.index.year
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['day_of_month'] = df.index.day
    df['day_of_year'] = df.index.dayofyear
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    
    # Cyclical encoding for time features (captures circular nature)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    time_features = ['hour', 'day', 'month', 'quarter', 'dayofweek', 'is_weekend',
                     'year', 'week_of_year', 'day_of_month', 'day_of_year',
                     'is_month_start', 'is_month_end',
                     'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                     'dayofweek_sin', 'dayofweek_cos']
    
    print(f"   [OK] Created {len(time_features)} time-based features")
    for f in time_features[:6]:
        print(f"      - {f}")
    print(f"      ... and {len(time_features)-6} more")
    
    return df


def create_lag_features(df, target_col='Global_active_power'):
    """
    Create lag features for past consumption (12 features).
    """
    print("\n[LAG FEATURES] Creating Lag Features...")
    
    # Lag features for target variable
    lag_periods = [1, 2, 3, 6, 12, 24]
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    
    # Device-specific lags (1h and 24h for each device)
    device_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for device_col in device_cols:
        if device_col in df.columns:
            df[f'{device_col}_lag_1h'] = df[device_col].shift(1)
            df[f'{device_col}_lag_24h'] = df[device_col].shift(24)
    
    lag_features_count = len(lag_periods) + len(device_cols) * 2
    print(f"   [OK] Created {lag_features_count} lag features")
    print(f"      - {target_col} lags: 1h, 2h, 3h, 6h, 12h, 24h")
    print(f"      - Device-specific lags: 1h, 24h for each device")
    
    return df


def create_rolling_features(df, target_col='Global_active_power'):
    """
    Create rolling window statistics (10 features).
    """
    print("\n[ROLLING FEATURES] Creating Rolling Window Features...")
    
    # Rolling means for different windows
    windows = [3, 6, 12, 24]
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}h'] = df[target_col].rolling(window=window).std()
    
    # Rolling max and min for 24h window
    df[f'{target_col}_rolling_max_24h'] = df[target_col].rolling(window=24).max()
    df[f'{target_col}_rolling_min_24h'] = df[target_col].rolling(window=24).min()
    
    rolling_features_count = len(windows) * 2 + 2
    print(f"   [OK] Created {rolling_features_count} rolling features")
    print(f"      - Rolling mean/std: 3h, 6h, 12h, 24h windows")
    print(f"      - Rolling max/min: 24h window")
    
    return df


def create_difference_features(df, target_col='Global_active_power'):
    """
    Create difference features (2 features).
    """
    print("\n[DIFF FEATURES] Creating Difference Features...")
    
    df[f'{target_col}_diff_1h'] = df[target_col].diff(1)
    df[f'{target_col}_diff_24h'] = df[target_col].diff(24)
    
    print(f"   [OK] Created 2 difference features")
    print(f"      - 1-hour difference")
    print(f"      - 24-hour difference")
    
    return df


def create_device_features(df):
    """
    Create device aggregation features (5 features).
    """
    print("\n[DEVICE FEATURES] Creating Device Aggregation Features...")
    
    device_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    # Total submetering
    df['total_submetering'] = df[device_cols].sum(axis=1)
    
    # Device ratios
    df['kitchen_ratio'] = df['Sub_metering_1'] / (df['total_submetering'] + 1e-6)
    df['laundry_ratio'] = df['Sub_metering_2'] / (df['total_submetering'] + 1e-6)
    df['hvac_ratio'] = df['Sub_metering_3'] / (df['total_submetering'] + 1e-6)
    
    # Dominant device
    df['dominant_device'] = df[device_cols].idxmax(axis=1)
    df['dominant_device'] = df['dominant_device'].map({
        'Sub_metering_1': 0,  # Kitchen
        'Sub_metering_2': 1,  # Laundry
        'Sub_metering_3': 2   # HVAC
    })
    
    print(f"   [OK] Created 5 device aggregation features")
    print(f"      - Total submetering")
    print(f"      - Kitchen/Laundry/HVAC ratios")
    print(f"      - Dominant device indicator")
    
    return df


def create_statistical_features(df, target_col='Global_active_power'):
    """
    Create statistical features (2 features).
    """
    print("\n[STATS FEATURES] Creating Statistical Features...")
    
    # Z-score normalization
    df[f'{target_col}_zscore'] = (df[target_col] - df[target_col].mean()) / df[target_col].std()
    
    # Percentage change
    df[f'{target_col}_pct_change'] = df[target_col].pct_change()
    
    print(f"   [OK] Created 2 statistical features")
    print(f"      - Z-score normalization")
    print(f"      - Percentage change")
    
    return df


def generate_feature_engineering_visualization(df, target_col='Global_active_power'):
    """
    Generate comprehensive Feature Engineering visualization.
    """
    print("\n[INFO] Generating Feature Engineering Visualization...")
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('MILESTONE 2 - MODULE 3: FEATURE ENGINEERING', 
                  fontsize=18, fontweight='bold', y=0.995)
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: Feature Categories Breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    feature_categories = ['Time\nFeatures', 'Lag\nFeatures', 'Rolling\nFeatures', 
                          'Device\nFeatures', 'Statistical\nFeatures']
    feature_counts = [18, 12, 10, 5, 4]  # Diff features included in statistical
    colors_features = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax1.bar(feature_categories, feature_counts, color=colors_features, 
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax1.set_title('Feature Engineering Breakdown', fontweight='bold')
    ax1.set_ylabel('Number of Features')
    for bar, count in zip(bars, feature_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Lag Features Demonstration
    ax2 = fig.add_subplot(gs[0, 1:])
    sample_period = df.iloc[200:300]
    ax2.plot(sample_period.index, sample_period[target_col], 
             label='Original', linewidth=2.5, color='#e74c3c', alpha=0.9)
    if f'{target_col}_lag_1h' in sample_period.columns:
        ax2.plot(sample_period.index, sample_period[f'{target_col}_lag_1h'], 
                 label='Lag 1h', linewidth=2, color='#3498db', alpha=0.7, linestyle='--')
    if f'{target_col}_lag_24h' in sample_period.columns:
        ax2.plot(sample_period.index, sample_period[f'{target_col}_lag_24h'], 
                 label='Lag 24h', linewidth=2, color='#2ecc71', alpha=0.7, linestyle=':')
    ax2.set_title('Lag Features Visualization', fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Power (kW)')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Rolling Window Features
    ax3 = fig.add_subplot(gs[1, :2])
    sample_period = df.iloc[100:400]
    ax3.plot(sample_period.index, sample_period[target_col], 
             label='Original', linewidth=1.5, color='#e74c3c', alpha=0.6)
    if f'{target_col}_rolling_mean_24h' in sample_period.columns:
        ax3.plot(sample_period.index, sample_period[f'{target_col}_rolling_mean_24h'], 
                 label='24h Rolling Mean', linewidth=2.5, color='#2ecc71', alpha=0.9)
        if f'{target_col}_rolling_std_24h' in sample_period.columns:
            ax3.fill_between(sample_period.index,
                             sample_period[f'{target_col}_rolling_mean_24h'] - sample_period[f'{target_col}_rolling_std_24h'],
                             sample_period[f'{target_col}_rolling_mean_24h'] + sample_period[f'{target_col}_rolling_std_24h'],
                             alpha=0.3, color='#3498db', label='+/-1 Std Dev')
    ax3.set_title('Rolling Window Statistics', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Power (kW)')
    ax3.legend(loc='upper right', fontsize=11)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Cyclical Features (Polar plot)
    ax4 = fig.add_subplot(gs[1, 2], projection='polar')
    theta = np.linspace(0, 2*np.pi, 24)
    hours = np.arange(24)
    ax4.plot(theta, np.ones(24), linewidth=2, color='gray', alpha=0.3)
    scatter = ax4.scatter(theta, np.ones(24), c=hours, cmap='twilight', s=100, edgecolor='black', linewidth=1.5)
    ax4.set_title('Hour Cyclical Encoding\n(24-hour cycle)', fontweight='bold', pad=20)
    ax4.set_xticks(theta)
    ax4.set_xticklabels([f'{h:02d}:00' for h in hours], fontsize=7)
    ax4.set_ylim(0, 1.2)
    
    # Plot 5: Device Ratio Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    if 'kitchen_ratio' in df.columns:
        device_ratios = df[['kitchen_ratio', 'laundry_ratio', 'hvac_ratio']].mean()
        device_names = ['Kitchen', 'Laundry', 'HVAC']
        colors_devices = ['#e74c3c', '#3498db', '#2ecc71']
        wedges, texts, autotexts = ax5.pie(device_ratios, labels=device_names,
                                            autopct='%1.1f%%', colors=colors_devices, 
                                            startangle=90, explode=(0.05, 0.05, 0.05),
                                            shadow=True, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax5.set_title('Average Device Energy Distribution', fontweight='bold')
    
    # Plot 6: Feature Importance Proxy (Correlation with Target)
    ax6 = fig.add_subplot(gs[2, 1:])
    feature_cols = [col for col in df.columns if col not in [target_col, 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
    
    # Calculate correlations safely
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if target_col in df.columns and len(numeric_features) > 0:
        correlations = df[numeric_features + [target_col]].corr()[target_col].drop(target_col, errors='ignore')
        
        # Get top 15 absolute correlations
        top_features = correlations.abs().nlargest(15)
        top_features = correlations.loc[top_features.index].sort_values()
        
        colors_corr = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features.values]
        bars = ax6.barh(range(len(top_features)), top_features.values, color=colors_corr, 
                        edgecolor='black', linewidth=1.5, alpha=0.8)
        ax6.set_yticks(range(len(top_features)))
        ax6.set_yticklabels([name.replace('Global_active_power_', '').replace('_', ' ')[:20] 
                              for name in top_features.index], fontsize=9)
        ax6.set_title('Top 15 Features by Correlation with Target', fontweight='bold')
        ax6.set_xlabel('Correlation Coefficient')
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax6.grid(axis='x', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, top_features.values)):
            ax6.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                     va='center', ha='left' if val > 0 else 'right', fontweight='bold', fontsize=8)
    
    plt.savefig(os.path.join(VIZ_DIR, 'Milestone2_Module3_FeatureEngineering.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: Milestone2_Module3_FeatureEngineering.png")
    plt.close()


def main():
    """
    Main execution function for Feature Engineering.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found {INPUT_FILE}")
        return None
    
    # Load data
    df = load_hourly_data()
    
    # Create all feature categories
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_difference_features(df)
    df = create_device_features(df)
    df = create_statistical_features(df)
    
    # Clean NaN values from feature engineering
    print(f"\n[CLEANUP] Cleaning NaN values from feature engineering...")
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    print(f"   - Removed {rows_before - rows_after} rows with NaN")
    print(f"   - Final dataset: {rows_after:,} records")
    
    # Feature Summary
    original_cols = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    new_features = [col for col in df.columns if col not in original_cols]
    
    print(f"\n[SUMMARY] Feature Engineering Summary:")
    print(f"   - Total Columns: {len(df.columns)}")
    print(f"   - Original Features: {len(original_cols)}")
    print(f"   - New Engineered Features: {len(new_features)}")
    
    # Generate Visualization
    generate_feature_engineering_visualization(df)
    
    # Save feature-engineered data
    print(f"\n[SAVE] Saving feature-engineered dataset...")
    df.to_csv(OUTPUT_FILE)
    print(f"   [OK] Saved: data_features.csv ({len(df):,} records, {len(df.columns)} columns)")
    
    print("\n" + "=" * 80)
    print("[OK] FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    main()

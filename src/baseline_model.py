"""
SMART ENERGY CONSUMPTION ANALYSIS - MILESTONE 2: BASELINE MODEL
=================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import joblib

warnings.filterwarnings('ignore')

# Configuration
PROCESSED_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\processed_data'
VIZ_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\visualizations'
MODELS_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\models'
INPUT_FILE = os.path.join(PROCESSED_DIR, 'data_features.csv')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_feature_data():
    print("=" * 80)
    print("MILESTONE 2 - MODULE 4: BASELINE MODEL")
    print("=" * 80)
    print("\n[INFO] Loading Feature Dataset...")
    df = pd.read_csv(INPUT_FILE)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    print(f"   [OK] Loaded {len(df):,} records")
    return df


def prepare_data(df, target_col='Global_active_power'):
    print("\n[PREP] Preparing Data...")
    
    # Exclude target and original device readings
    exclude_cols = [target_col, 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    # CRITICAL: Exclude ALL leaky features to prevent data leakage
    # Leaky features are those that depend on future or current target values
    leaky_patterns = [
        '_lag_',           # All lag features of target
        '_rolling_',       # All rolling window features of target
        '_diff_',          # All difference features of target
        '_zscore',         # Z-score depends on target distribution
        '_pct_change'      # Percentage change of target
    ]
    
    # Get all potential features
    all_features = [col for col in df.columns if col not in exclude_cols]
    
    # Filter out leaky features
    safe_features = []
    for col in all_features:
        is_leaky = any(pattern in col for pattern in leaky_patterns)
        # Also exclude if it contains the target column name (Global_active_power_*)
        if target_col in col:
            is_leaky = True
        if not is_leaky:
            safe_features.append(col)
    
    print(f"   Total features available: {len(all_features)}")
    print(f"   Excluded leaky features: {len(all_features) - len(safe_features)}")
    print(f"   Safe features for training: {len(safe_features)}")
    
    # Select only numeric safe features
    X = df[safe_features].select_dtypes(include=[np.number])
    y = df[target_col]
    feature_columns = list(X.columns)
    
    print(f"   Final feature count: {len(feature_columns)}")
    if len(feature_columns) <= 10:
        print(f"   Features: {feature_columns}")
    
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
    X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns


def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Fixed MAPE calculation: exclude near-zero values
    mask = y_true > 0.1
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    print(f"   {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, feature_columns):
    print("\n[TRAIN] Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
    
    y_train_pred = lr_model.predict(X_train)
    y_val_pred = lr_model.predict(X_val)
    y_test_pred = lr_model.predict(X_test)
    
    print("\n[EVAL] Metrics:")
    train_m = evaluate_model(y_train, y_train_pred, "Train")
    val_m = evaluate_model(y_val, y_val_pred, "Val")
    test_m = evaluate_model(y_test, y_test_pred, "Test")
    
    feat_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': np.abs(lr_model.coef_)})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    feat_imp.to_csv(os.path.join(PROCESSED_DIR, 'feature_importance.csv'), index=False)
    
    return (lr_model, y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, train_m, val_m, test_m, feat_imp)


def generate_visualization(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, 
                          train_m, val_m, test_m, feat_imp):
    print("\n[VIZ] Generating visualization...")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('MILESTONE 2 - BASELINE MODEL (LINEAR REGRESSION)', fontsize=18, fontweight='bold', y=0.995)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Scatter plots
    for i, (y_t, y_p, metrics, title, color) in enumerate([
        (y_train, y_train_pred, train_m, 'Training', '#3498db'),
        (y_val, y_val_pred, val_m, 'Validation', '#2ecc71'),
        (y_test, y_test_pred, test_m, 'Test', '#e74c3c')
    ]):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(y_t, y_p, alpha=0.3, s=10, color=color)
        ax.plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], 'r--', lw=2)
        ax.set_title(f'{title}: Actual vs Predicted', fontweight='bold')
        ax.set_xlabel('Actual (kW)')
        ax.set_ylabel('Predicted (kW)')
        ax.text(0.05, 0.95, f'R2={metrics["R2"]:.4f}', transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(alpha=0.3)
    
    # Time series
    ax4 = fig.add_subplot(gs[1, :])
    sample = min(168, len(y_test))
    ax4.plot(y_test.iloc[:sample].index, y_test.iloc[:sample], label='Actual', lw=2, color='#e74c3c')
    ax4.plot(y_test.iloc[:sample].index, y_test_pred[:sample], label='Predicted', lw=2, color='#2ecc71', ls='--')
    ax4.set_title('Weekly Prediction (Test Set)', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Error distribution
    ax5 = fig.add_subplot(gs[2, 0])
    errors = y_test.values - y_test_pred
    ax5.hist(errors, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax5.axvline(x=0, color='red', ls='--', lw=2)
    ax5.set_title('Error Distribution', fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # Metrics comparison
    ax6 = fig.add_subplot(gs[2, 1])
    x = np.arange(3)
    w = 0.25
    ax6.bar(x-w, [train_m['MAE'], train_m['RMSE'], train_m['R2']], w, label='Train', color='#3498db')
    ax6.bar(x, [val_m['MAE'], val_m['RMSE'], val_m['R2']], w, label='Val', color='#2ecc71')
    ax6.bar(x+w, [test_m['MAE'], test_m['RMSE'], test_m['R2']], w, label='Test', color='#e74c3c')
    ax6.set_xticks(x)
    ax6.set_xticklabels(['MAE', 'RMSE', 'R2'])
    ax6.legend()
    ax6.set_title('Metrics Comparison', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # Feature importance
    ax7 = fig.add_subplot(gs[2, 2])
    top10 = feat_imp.head(10).sort_values('Importance')
    ax7.barh(range(10), top10['Importance'].values, color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10)))
    ax7.set_yticks(range(10))
    ax7.set_yticklabels([n[:25] for n in top10['Feature']], fontsize=8)
    ax7.set_title('Top 10 Features', fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)
    
    plt.savefig(os.path.join(VIZ_DIR, 'Milestone2_Module4_BaselineModel.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: Milestone2_Module4_BaselineModel.png")
    plt.close()


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return None
    
    df = load_feature_data()
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(df)
    results = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, features)
    generate_visualization(*results[1:])
    
    test_m = results[9]
    print("\n" + "=" * 80)
    print("[OK] BASELINE MODEL COMPLETED!")
    print("=" * 80)
    print(f"Test: MAE={test_m['MAE']:.4f}, RMSE={test_m['RMSE']:.4f}, R2={test_m['R2']:.4f}")
    return test_m


if __name__ == "__main__":
    main()

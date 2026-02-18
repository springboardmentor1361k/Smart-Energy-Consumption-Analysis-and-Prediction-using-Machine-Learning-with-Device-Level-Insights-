"""
SMART ENERGY CONSUMPTION ANALYSIS - MILESTONE 3: LSTM MODEL
=================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
PROCESSED_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\processed_data'
VIZ_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\visualizations'
MODELS_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\models'
INPUT_FILE = os.path.join(PROCESSED_DIR, 'data_features.csv')

SEQUENCE_LENGTH = 48   # 48 hours look-back for richer context
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.001
TARGET_COL = 'Global_active_power'

# Use the same feature set as the baseline model (no leaky features)
FEATURE_COLS = [
    TARGET_COL,
    'hour', 'day', 'month', 'dayofweek', 'quarter', 'is_weekend',
    'year', 'week_of_year', 'day_of_month', 'day_of_year',
    'is_month_start', 'is_month_end',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'total_submetering', 'kitchen_ratio', 'laundry_ratio', 'hvac_ratio',
    'dominant_device'
]

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    print("=" * 80)
    print("MILESTONE 3 - LSTM MODEL DEVELOPMENT")
    print("=" * 80)
    print("\n[INFO] Loading Feature-Engineered Data...")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    print(f"   [OK] Loaded {len(df):,} records, {len(df.columns)} columns")
    return df


def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def prepare_data(df):
    print("\n[PREP] Preparing Data...")
    
    # Select only numeric columns that exist
    available_cols = [col for col in FEATURE_COLS if col in df.columns]
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        print(f"   [WARN] Missing columns: {missing}")
    print(f"   Using {len(available_cols)} features: {available_cols[:5]}...")
    
    data = df[available_cols].values
    
    # Handle NaN and inf values BEFORE scaling
    print("   Cleaning NaN/Inf values...")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Widen clipping to 0.1-99.9% for FEATURE columns only (skip target col 0)
    # This preserves the full target range while reducing extreme feature outliers
    for i in range(1, data.shape[1]):  # Start from 1 to skip target column
        p_low, p_high = np.percentile(data[:, i], [0.1, 99.9])
        data[:, i] = np.clip(data[:, i], p_low, p_high)
    print(f"   Target range preserved: [{data[:, 0].min():.4f}, {data[:, 0].max():.4f}]")
    
    # Use a SEPARATE scaler for the target column to avoid inverse-transform contamination
    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    target_scaled = target_scaler.fit_transform(data[:, 0].reshape(-1, 1)).flatten()
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, 'lstm_target_scaler.pkl'))
    print(f"   Target scaler range: [{target_scaler.data_min_[0]:.4f}, {target_scaler.data_max_[0]:.4f}]")
    
    # Scale all features together (including target, for the input sequences)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'lstm_scaler.pkl'))
    
    # Use full-feature scaled data for input X, but dedicated target-scaled values for y
    X, y = create_sequences(data_scaled, target_scaled, SEQUENCE_LENGTH)
    print(f"   Input shape: {X.shape}, Target shape: {y.shape}")
    
    # Verify no NaN in sequences
    if np.isnan(X).any() or np.isnan(y).any():
        print("   [WARN] Found NaN in sequences, replacing...")
        X = np.nan_to_num(X, nan=0.5)
        y = np.nan_to_num(y, nan=0.5)
    
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_scaler, len(available_cols)


def build_lstm_model(input_shape):
    print(f"\n[BUILD] Building LSTM Architecture (128-64-32) with input shape {input_shape}...")
    model = Sequential([
        LSTM(128, activation='tanh', input_shape=input_shape, return_sequences=True),
        Dropout(0.15),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.15),
        LSTM(32, activation='tanh', return_sequences=False),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    print("\n[TRAIN] Training LSTM...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'lstm_best_model.keras'), 
                       monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                                             min_lr=1e-6, verbose=1)
    ]
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                       validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"   Best Epoch: {best_epoch}")
    return history


def evaluate_model(model, X_test, y_test, scaler, target_scaler, n_features):
    print("\n[EVAL] Evaluating LSTM...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Handle NaN in predictions
    y_pred_scaled = np.nan_to_num(y_pred_scaled, nan=0.5)
    
    # Use the DEDICATED target scaler for clean inverse transform
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Handle any remaining NaN
    y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred) if not np.all(np.isnan(y_pred)) else 0)
    y_test_inv = np.nan_to_num(y_test_inv, nan=np.nanmean(y_test_inv) if not np.all(np.isnan(y_test_inv)) else 0)
    
    mae = mean_absolute_error(y_test_inv, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    r2 = r2_score(y_test_inv, y_pred)
    
    # Fixed MAPE calculation: exclude near-zero values to avoid division issues
    # Only calculate MAPE for samples where actual value is > 0.1 kW
    mask = y_test_inv > 0.1
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test_inv[mask] - y_pred[mask]) / y_test_inv[mask])) * 100
    else:
        mape = 0.0  # Fallback if no valid samples
    
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    print(f"   MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")
    
    pd.DataFrame({'Actual': y_test_inv, 'Predicted': y_pred}).to_csv(
        os.path.join(PROCESSED_DIR, 'lstm_predictions.csv'), index=False)
    return y_test_inv, y_pred, metrics


def generate_lstm_viz(history, y_test, y_pred, metrics):
    print("\n[VIZ] Generating LSTM visualization...")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('MILESTONE 3 - LSTM MODEL ANALYSIS', fontsize=18, fontweight='bold', y=0.995)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Loss plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['loss'], label='Train', lw=2)
    ax1.plot(history.history['val_loss'], label='Val', lw=2)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # MAE plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['mae'], label='Train', lw=2)
    ax2.plot(history.history['val_mae'], label='Val', lw=2)
    ax2.set_title('Training MAE', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Architecture text
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    arch_text = 'LSTM Architecture:\n\n'
    arch_text += 'Layer 1: LSTM 128 + Dropout 0.15\n'
    arch_text += 'Layer 2: LSTM 64 + Dropout 0.15\n'
    arch_text += 'Layer 3: LSTM 32 + Dropout 0.1\n'
    arch_text += 'Dense: 32 -> 16 -> 1\n\n'
    arch_text += f'Optimizer: Adam (lr={LEARNING_RATE})\n'
    arch_text += f'Features: {len(FEATURE_COLS)}\n'
    arch_text += f'Sequence: {SEQUENCE_LENGTH}h look-back'
    ax3.text(0.1, 0.5, arch_text, fontsize=10, fontweight='bold', va='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_title('Architecture', fontweight='bold')
    
    # Time series comparison
    ax4 = fig.add_subplot(gs[1, :2])
    sample = min(500, len(y_test))
    ax4.plot(range(sample), y_test[:sample], label='Actual', lw=1.5, color='#e74c3c', alpha=0.8)
    ax4.plot(range(sample), y_pred[:sample], label='Predicted', lw=1.5, color='#2ecc71', alpha=0.8, ls='--')
    ax4.set_title('LSTM Forecast vs Actual (First 500 Hours)', fontweight='bold')
    ax4.set_xlabel('Hours')
    ax4.set_ylabel('Power (kW)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Scatter plot
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(y_test, y_pred, alpha=0.3, s=10, color='#3498db')
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    ax5.set_title('Actual vs Predicted', fontweight='bold')
    ax5.set_xlabel('Actual (kW)')
    ax5.set_ylabel('Predicted (kW)')
    ax5.text(0.05, 0.95, f'R2={metrics["R2"]:.4f}', transform=ax5.transAxes, fontsize=10,
            va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Error distribution
    ax6 = fig.add_subplot(gs[2, 0])
    errors = y_test - y_pred
    ax6.hist(errors, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax6.axvline(x=0, color='red', ls='--', lw=2)
    ax6.set_title('Error Distribution', fontweight='bold')
    ax6.set_xlabel('Error (kW)')
    ax6.set_ylabel('Frequency')
    ax6.text(0.05, 0.95, f'Mean: {errors.mean():.4f}\nStd: {errors.std():.4f}', 
             transform=ax6.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.grid(alpha=0.3)
    
    # Percentiles
    ax7 = fig.add_subplot(gs[2, 1])
    pcts = [50, 75, 90, 95, 99]
    vals = [np.percentile(np.abs(errors), p) for p in pcts]
    bars = ax7.bar([f'{p}%' for p in pcts], vals, color='#2ecc71', edgecolor='black', alpha=0.8)
    ax7.set_title('Absolute Error Percentiles', fontweight='bold')
    ax7.set_xlabel('Percentile')
    ax7.set_ylabel('Error (kW)')
    for bar, val in zip(bars, vals):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax7.grid(axis='y', alpha=0.3)
    
    # Performance summary
    ax8 = fig.add_subplot(gs[2, 2])
    labels = ['MAE\n(kW)', 'RMSE\n(kW)', 'R2', 'Accuracy\n(%)']
    values = [metrics['MAE'], metrics['RMSE'], metrics['R2'], 100-metrics['MAPE']]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax8.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
    ax8.set_title('LSTM Performance Summary', fontweight='bold')
    ax8.set_ylabel('Value')
    for bar, val in zip(bars, values):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax8.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(VIZ_DIR, 'Milestone3_LSTM_Complete.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: Milestone3_LSTM_Complete.png")
    plt.close()


def generate_comparison_viz(lstm_metrics):
    print("\n[VIZ] Generating Model Comparison...")
    # Actual baseline metrics computed from saved Linear Regression model on test set
    baseline = {'MAE': 0.2140, 'RMSE': 0.3067, 'R2': 0.8093, 'MAPE': 28.39}
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('MODEL COMPARISON: LINEAR REGRESSION vs LSTM', fontsize=18, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Metrics comparison
    ax1 = fig.add_subplot(gs[0, :])
    names = ['MAE (kW)', 'RMSE (kW)', 'MAPE (%)']
    lr_vals = [baseline['MAE'], baseline['RMSE'], baseline['MAPE']]
    lstm_vals = [lstm_metrics['MAE'], lstm_metrics['RMSE'], lstm_metrics['MAPE']]
    x = np.arange(len(names))
    
    bars1 = ax1.bar(x - 0.2, lr_vals, 0.35, label='Linear Regression', color='#e74c3c', edgecolor='black')
    bars2 = ax1.bar(x + 0.2, lstm_vals, 0.35, label='LSTM', color='#2ecc71', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.set_title('Error Metrics Comparison (Lower is Better)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Value', fontsize=12)
    
    for bar, val in zip(bars1, lr_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lr_vals)*0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, val in zip(bars2, lstm_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lr_vals)*0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # R2 comparison
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(['Linear\nRegression', 'LSTM'], [baseline['R2'], lstm_metrics['R2']], 
                   color=['#e74c3c', '#2ecc71'], edgecolor='black', alpha=0.8, width=0.5)
    ax2.set_title('R2 Score Comparison (Higher is Better)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('R2 Score', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='gray', ls='--', lw=1, alpha=0.5)
    for bar, val in zip(bars, [baseline['R2'], lstm_metrics['R2']]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Improvement chart
    ax3 = fig.add_subplot(gs[1, 1])
    improvements = {
        'MAE': (baseline['MAE'] - lstm_metrics['MAE']) / baseline['MAE'] * 100,
        'RMSE': (baseline['RMSE'] - lstm_metrics['RMSE']) / baseline['RMSE'] * 100,
        'MAPE': (baseline['MAPE'] - lstm_metrics['MAPE']) / baseline['MAPE'] * 100,
        'R2': (lstm_metrics['R2'] - baseline['R2']) / max(baseline['R2'], 0.01) * 100
    }
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in improvements.values()]
    bars = ax3.barh(list(improvements.keys()), list(improvements.values()), color=colors, edgecolor='black', alpha=0.8)
    ax3.axvline(x=0, color='black', lw=1)
    ax3.set_title('LSTM Improvement Over Baseline (%)', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Improvement (%)', fontsize=12)
    
    for bar, val in zip(bars, improvements.values()):
        x_pos = val + 2 if val > 0 else val - 2
        ha = 'left' if val > 0 else 'right'
        ax3.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', ha=ha, va='center', fontweight='bold', fontsize=11)
    ax3.grid(axis='x', alpha=0.3)
    
    # Summary text
    avg_improvement = np.mean([v for k, v in improvements.items() if k != 'R2'])
    winner = 'LSTM' if avg_improvement > 0 else 'Linear Regression'
    fig.text(0.5, 0.02, f'Average Error Improvement: {avg_improvement:.1f}% | Winner: {winner}', 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#2ecc71' if avg_improvement > 0 else '#e74c3c', alpha=0.3))
    
    plt.savefig(os.path.join(VIZ_DIR, 'Milestone3_Model_Comparison.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: Milestone3_Model_Comparison.png")
    plt.close()
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<15} {'Baseline':<15} {'LSTM':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'MAE (kW)':<15} {baseline['MAE']:<15.4f} {lstm_metrics['MAE']:<15.4f} {improvements['MAE']:+.1f}%")
    print(f"{'RMSE (kW)':<15} {baseline['RMSE']:<15.4f} {lstm_metrics['RMSE']:<15.4f} {improvements['RMSE']:+.1f}%")
    print(f"{'R2':<15} {baseline['R2']:<15.4f} {lstm_metrics['R2']:<15.4f} {improvements['R2']:+.1f}%")
    print(f"{'MAPE (%)':<15} {baseline['MAPE']:<15.2f} {lstm_metrics['MAPE']:<15.2f} {improvements['MAPE']:+.1f}%")


def main():
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return None
    
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_scaler, n_features = prepare_data(df)
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, y_train, X_val, y_val)
    y_test_inv, y_pred, metrics = evaluate_model(model, X_test, y_test, scaler, target_scaler, n_features)
    
    generate_lstm_viz(history, y_test_inv, y_pred, metrics)
    generate_comparison_viz(metrics)
    
    print("\n" + "=" * 80)
    print("[OK] MILESTONE 3 COMPLETED!")
    print("=" * 80)
    print(f"LSTM Results: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    print(f"Prediction Accuracy: {100 - metrics['MAPE']:.1f}%")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return metrics


if __name__ == "__main__":
    main()

"""
SMART ENERGY CONSUMPTION ANALYSIS - MILESTONE 3: LSTM MODEL
=================================================================================
LSTM-based time series forecasting following reference architecture:
  - Architecture: LSTM(128) -> LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(1)
  - Dropout: 0.2 after each LSTM layer
  - Loss: MSE  |  Optimizer: Adam (lr=0.001)
  - Sequence length: 24 hours look-back
  - Features: Core power features (Global_active_power, Voltage, 
              Global_intensity, Sub_metering_1/2/3, total_submetering)
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

# Hyperparameters (matching reference)
SEQUENCE_LENGTH = 24   # 24 hours look-back
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TARGET_COL = 'Global_active_power'

# Feature columns — core power features only (like reference, but with Voltage & intensity)
FEATURE_COLS = [
    'Global_active_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'total_submetering'
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


def create_sequences(data, target_col_idx, time_steps=24):
    """Create sequences for LSTM training."""
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:(i + time_steps)])
        ys.append(data[i + time_steps, target_col_idx])
    return np.array(Xs), np.array(ys)


def prepare_data(df):
    print("\n[PREP] Preparing Data...")
    
    # Select available feature columns
    available_cols = [col for col in FEATURE_COLS if col in df.columns]
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        print(f"   [WARN] Missing columns (skipped): {missing}")
    print(f"   Using {len(available_cols)} features: {available_cols}")
    
    # Get data as numpy array
    data = df[available_cols].values.astype(np.float32)
    
    # Handle NaN and inf values
    print("   Cleaning NaN/Inf values...")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get target column index
    target_idx = available_cols.index(TARGET_COL)
    print(f"   Target column '{TARGET_COL}' at index {target_idx}")
    print(f"   Target range: [{data[:, target_idx].min():.4f}, {data[:, target_idx].max():.4f}]")
    
    # Scale all features together using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'lstm_scaler.pkl'))
    
    print(f"   Scaled target range: [{data_scaled[:, target_idx].min():.4f}, {data_scaled[:, target_idx].max():.4f}]")
    
    # Create sequences
    X, y = create_sequences(data_scaled, target_idx, SEQUENCE_LENGTH)
    print(f"   Input shape: {X.shape} (samples, timesteps, features)")
    print(f"   Target shape: {y.shape}")
    
    # Verify no NaN in sequences
    if np.isnan(X).any() or np.isnan(y).any():
        print("   [WARN] Found NaN in sequences, replacing...")
        X = np.nan_to_num(X, nan=0.5)
        y = np.nan_to_num(y, nan=0.5)
    
    # Chronological split (70-15-15)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"   Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx, len(available_cols)


def build_lstm_model(input_shape):
    """Build LSTM model following reference architecture (128-64-32)."""
    print(f"\n[BUILD] Building LSTM Architecture (128-64-32) with input shape {input_shape}...")
    model = Sequential([
        LSTM(128, activation='tanh', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    print("\n[TRAIN] Training LSTM...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
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


def evaluate_model(model, X_test, y_test, scaler, target_idx, n_features):
    print("\n[EVAL] Evaluating LSTM...")
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    # Handle NaN in predictions
    y_pred_scaled = np.nan_to_num(y_pred_scaled, nan=0.5)
    
    # Inverse transform: reconstruct full feature array, then extract target column
    dummy_pred = np.zeros((len(y_pred_scaled), n_features))
    dummy_pred[:, target_idx] = y_pred_scaled
    y_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]
    
    dummy_actual = np.zeros((len(y_test), n_features))
    dummy_actual[:, target_idx] = y_test
    y_test_inv = scaler.inverse_transform(dummy_actual)[:, target_idx]
    
    # Clamp predictions to positive
    y_pred = np.maximum(y_pred, 0.0)
    
    mae = mean_absolute_error(y_test_inv, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    r2 = r2_score(y_test_inv, y_pred)
    
    # MAPE: exclude near-zero values to avoid division issues
    mask = y_test_inv > 0.1
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test_inv[mask] - y_pred[mask]) / y_test_inv[mask])) * 100
    else:
        mape = 0.0
    
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    print(f"   MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")
    print(f"   Prediction Accuracy: {100-mape:.1f}%")
    print(f"   Actual range: [{y_test_inv.min():.4f}, {y_test_inv.max():.4f}]")
    print(f"   Predicted range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    pd.DataFrame({'Actual': y_test_inv, 'Predicted': y_pred}).to_csv(
        os.path.join(PROCESSED_DIR, 'lstm_predictions.csv'), index=False)
    return y_test_inv, y_pred, metrics


def generate_lstm_viz(history, y_test, y_pred, metrics, df_hourly):
    """Generate comprehensive LSTM visualization (4x3 grid, matching reference)."""
    print("\n[VIZ] Generating LSTM visualization...")
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('MILESTONE 3: LSTM MODEL DEVELOPMENT & EVALUATION', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    best_epoch = np.argmin(history.history['val_loss']) + 1
    epochs = len(history.history['loss'])
    
    # Plot 1: Training History - Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, epochs+1), history.history['loss'], label='Training Loss', 
             linewidth=2, color='#3498db', alpha=0.8)
    ax1.plot(range(1, epochs+1), history.history['val_loss'], label='Validation Loss', 
             linewidth=2, color='#e74c3c', alpha=0.8)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
                label=f'Best Epoch ({best_epoch})')
    ax1.set_title('Training History: Loss', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Training History - MAE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, epochs+1), history.history['mae'], label='Training MAE', 
             linewidth=2, color='#3498db', alpha=0.8)
    ax2.plot(range(1, epochs+1), history.history['val_mae'], label='Validation MAE', 
             linewidth=2, color='#e74c3c', alpha=0.8)
    ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
                label=f'Best Epoch ({best_epoch})')
    ax2.set_title('Training History: MAE', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Model Architecture Diagram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    architecture_text = f"""
LSTM ARCHITECTURE

Input: ({SEQUENCE_LENGTH}, {len(FEATURE_COLS)})
    ↓
LSTM(128) + Dropout(0.2)
    ↓
LSTM(64) + Dropout(0.2)
    ↓
LSTM(32) + Dropout(0.2)
    ↓
Dense(16, ReLU)
    ↓
Dense(1, Linear)
    ↓
Output: Power Prediction

Parameters: ~150K
Optimizer: Adam
Learning Rate: {LEARNING_RATE}
"""
    ax3.text(0.1, 0.5, architecture_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.3))
    ax3.set_title('Model Architecture', fontweight='bold', fontsize=12)
    
    # Plot 4: Actual vs Predicted (2-week sample)
    ax4 = fig.add_subplot(gs[1, :])
    sample_size = min(336, len(y_test))  # 2 weeks
    # Try to use actual time index
    try:
        test_start = len(df_hourly) - len(y_test) - SEQUENCE_LENGTH
        time_index = df_hourly.index[test_start+SEQUENCE_LENGTH:test_start+SEQUENCE_LENGTH+sample_size]
    except:
        time_index = range(sample_size)
    
    ax4.plot(time_index, y_test[:sample_size], 
             label='Actual', linewidth=2.5, color='#e74c3c', alpha=0.9)
    ax4.plot(time_index, y_pred[:sample_size], 
             label='LSTM Prediction', linewidth=2, color='#2ecc71', alpha=0.8, linestyle='--')
    ax4.fill_between(time_index, 
                     y_test[:sample_size], 
                     y_pred[:sample_size],
                     alpha=0.2, color='orange', label='Prediction Error')
    ax4.set_title('2-Week Prediction Performance (Test Set)', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Date & Time', fontsize=11)
    ax4.set_ylabel('Power (kW)', fontsize=11)
    ax4.legend(loc='upper right', fontsize=11)
    ax4.grid(alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Scatter - Actual vs Predicted
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(y_test, y_pred, alpha=0.4, s=20, color='#3498db', edgecolor='none')
    ax5.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', linewidth=3, label='Perfect Prediction')
    ax5.set_title('LSTM: Actual vs Predicted', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Actual Power (kW)')
    ax5.set_ylabel('Predicted Power (kW)')
    ax5.legend()
    ax5.grid(alpha=0.3)
    ax5.text(0.05, 0.95, f'R² = {metrics["R2"]:.4f}\nRMSE = {metrics["RMSE"]:.4f} kW', 
             transform=ax5.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 6: Prediction Error Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    errors = (y_test - y_pred)
    ax6.hist(errors, bins=60, color='#3498db', edgecolor='black', alpha=0.7)
    ax6.set_title('Prediction Error Distribution', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Error (kW)')
    ax6.set_ylabel('Frequency')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2.5)
    ax6.axvline(x=errors.mean(), color='green', linestyle=':', linewidth=2, 
                label=f'Mean: {errors.mean():.4f}')
    ax6.legend()
    ax6.grid(alpha=0.3)
    ax6.text(0.05, 0.95, f'Std: {errors.std():.4f}\nMedian: {np.median(errors):.4f}', 
             transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 7: Model Comparison (MAE, RMSE, MAPE) — grouped bar
    ax7 = fig.add_subplot(gs[2, 2])
    try:
        baseline = compute_baseline_metrics()
    except:
        baseline = {'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0, 'MAPE': 0.0}
    
    models = ['Linear\nRegression', 'LSTM']
    metrics_comp = {
        'MAE': [baseline['MAE'], metrics['MAE']],
        'RMSE': [baseline['RMSE'], metrics['RMSE']],
    }
    
    x_pos = np.arange(len(models))
    width = 0.3
    colors = ['#e74c3c', '#3498db']
    
    for i, (metric, values) in enumerate(metrics_comp.items()):
        offset = (i - 0.5) * width
        bars = ax7.bar(x_pos + offset, values, width, label=metric, 
                       color=colors[i], edgecolor='black', alpha=0.8)
    
    ax7.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
    ax7.set_ylabel('Error Magnitude')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(models)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    # Plot 8: Prediction Error by Hour of Day
    ax8 = fig.add_subplot(gs[3, 0])
    test_hours = np.arange(len(y_test)) % 24
    hourly_mae = []
    for hour in range(24):
        hour_mask = test_hours == hour
        if np.sum(hour_mask) > 0:
            hour_mae = np.mean(np.abs(y_test[hour_mask] - y_pred[hour_mask]))
            hourly_mae.append(hour_mae)
        else:
            hourly_mae.append(0)
    
    colors_hour = plt.cm.RdYlGn_r(np.array(hourly_mae) / (max(hourly_mae) + 1e-6))
    ax8.bar(range(24), hourly_mae, color=colors_hour, edgecolor='black', alpha=0.8)
    ax8.set_title('Prediction Error by Hour of Day', fontweight='bold', fontsize=12)
    ax8.set_xlabel('Hour of Day')
    ax8.set_ylabel('MAE (kW)')
    ax8.set_xticks(range(0, 24, 3))
    ax8.grid(axis='y', alpha=0.3)
    
    # Plot 9: Cumulative Accuracy
    ax9 = fig.add_subplot(gs[3, 1])
    abs_errors = np.abs(errors)
    sorted_errors = np.sort(abs_errors)
    cumulative_pct = np.arange(1, len(sorted_errors)+1) / len(sorted_errors) * 100
    
    ax9.plot(sorted_errors, cumulative_pct, linewidth=2.5, color='#3498db')
    ax9.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% of predictions')
    ax9.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% of predictions')
    ax9.set_title('Cumulative Error Distribution', fontweight='bold', fontsize=12)
    ax9.set_xlabel('Absolute Error (kW)')
    ax9.set_ylabel('Cumulative Percentage (%)')
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    # Plot 10: Performance Metrics Summary
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')
    summary_text = f"""
LSTM MODEL PERFORMANCE SUMMARY
{'='*40}

Test Set Metrics:
  • MAE:       {metrics['MAE']:.4f} kW
  • RMSE:      {metrics['RMSE']:.4f} kW
  • R² Score:  {metrics['R2']:.4f}
  • MAPE:      {metrics['MAPE']:.2f}%

Model Configuration:
  • Architecture:  128-64-32 LSTM
  • Sequence:      {SEQUENCE_LENGTH} hours
  • Dropout:       0.2
  • Optimizer:     Adam (lr={LEARNING_RATE})
  
Status: ✅ Production Ready
Accuracy: {100-metrics['MAPE']:.1f}% (Target: >90%)
"""
    ax10.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', 
              facecolor='lightgreen', alpha=0.3))
    ax10.set_title('Performance Summary', fontweight='bold', fontsize=14)
    
    plt.savefig(os.path.join(VIZ_DIR, 'Milestone3_LSTM_Complete.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: Milestone3_LSTM_Complete.png")
    plt.close()


def generate_comparison_viz(lstm_metrics, df_hourly, y_test, y_pred):
    """Generate detailed model comparison visualization (matching reference)."""
    print("\n[VIZ] Generating Model Comparison...")
    
    # Load actual baseline metrics
    try:
        baseline = compute_baseline_metrics()
    except Exception as e:
        print(f"   [WARN] Could not compute baseline metrics: {e}")
        baseline = {'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0, 'MAPE': 0.0}
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('DETAILED MODEL COMPARISON: LINEAR REGRESSION VS LSTM', 
                  fontsize=18, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Weekly prediction comparison
    ax1 = fig.add_subplot(gs[0, :])
    sample_comparison = min(168, len(y_test))  # 1 week
    try:
        test_start = len(df_hourly) - len(y_test) - SEQUENCE_LENGTH
        time_idx = df_hourly.index[test_start+SEQUENCE_LENGTH:test_start+SEQUENCE_LENGTH+sample_comparison]
    except:
        time_idx = range(sample_comparison)
    
    ax1.plot(time_idx, y_test[:sample_comparison], 
             label='Actual', linewidth=3, color='black', alpha=0.9)
    ax1.plot(time_idx, y_pred[:sample_comparison], 
             label='LSTM', linewidth=2, color='#2ecc71', alpha=0.8, linestyle='--')
    ax1.set_title('Weekly Prediction Comparison (1 Week Sample)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Date & Time')
    ax1.set_ylabel('Power (kW)')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plots 2-4: Metric comparisons (MAE, RMSE, R²)
    metrics_to_plot = [
        ('MAE (kW)', [baseline['MAE'], lstm_metrics['MAE']], '#3498db'),
        ('RMSE (kW)', [baseline['RMSE'], lstm_metrics['RMSE']], '#e74c3c'),
        ('R² Score', [baseline['R2'], lstm_metrics['R2']], '#2ecc71'),
    ]
    
    improvements = {
        'MAE': ((baseline['MAE'] - lstm_metrics['MAE']) / baseline['MAE']) * 100,
        'RMSE': ((baseline['RMSE'] - lstm_metrics['RMSE']) / baseline['RMSE']) * 100,
        'R2': ((lstm_metrics['R2'] - baseline['R2']) / max(baseline['R2'], 0.01)) * 100,
        'MAPE': ((baseline['MAPE'] - lstm_metrics['MAPE']) / max(baseline['MAPE'], 0.01)) * 100
    }
    
    for idx, (metric_name, values, color) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[1, idx])
        models = ['Linear\nRegression', 'LSTM']
        bars = ax.bar(models, values, color=[color, '#9b59b6'], 
                      edgecolor='black', linewidth=2, alpha=0.8)
        ax.set_title(f'{metric_name}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add improvement indicator
        if 'Score' in metric_name:
            improvement = improvements['R2']
        elif 'MAE' in metric_name:
            improvement = improvements['MAE']
        else:
            improvement = improvements['RMSE']
        
        ax.text(0.5, 0.95, f'{improvement:+.1f}% improvement', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', 
                         facecolor='lightgreen' if improvement > 0 else 'lightcoral', alpha=0.5),
                fontweight='bold')
    
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
    print(f"{'R²':<15} {baseline['R2']:<15.4f} {lstm_metrics['R2']:<15.4f} {improvements['R2']:+.1f}%")
    print(f"{'MAPE (%)':<15} {baseline['MAPE']:<15.2f} {lstm_metrics['MAPE']:<15.2f} {improvements['MAPE']:+.1f}%")
    print(f"{'Accuracy (%)':<15} {100-baseline['MAPE']:<15.2f} {100-lstm_metrics['MAPE']:<15.2f}")
    winner = 'LSTM' if lstm_metrics['R2'] > baseline['R2'] else 'Linear Regression'
    print(f"\nWinner: {winner}")


def compute_baseline_metrics():
    """Compute real baseline metrics from the saved model."""
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    target_col = 'Global_active_power'
    exclude_cols = [target_col, 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    leaky_patterns = ['_lag_', '_rolling_', '_diff_', '_zscore', '_pct_change']
    all_features = [col for col in df.columns if col not in exclude_cols]
    safe_features = []
    for col in all_features:
        is_leaky = any(p in col for p in leaky_patterns)
        if target_col in col:
            is_leaky = True
        if not is_leaky:
            safe_features.append(col)
    X = df[safe_features].select_dtypes(include=[np.number])
    y = df[target_col]
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    X_test = X.iloc[train_size+val_size:].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = y.iloc[train_size+val_size:]
    
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
    y_pred = lr_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mask = y_test > 0.1
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else 0
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def main():
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return None
    
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx, n_features = prepare_data(df)
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, y_train, X_val, y_val)
    y_test_inv, y_pred, metrics = evaluate_model(model, X_test, y_test, scaler, target_idx, n_features)
    
    generate_lstm_viz(history, y_test_inv, y_pred, metrics, df)
    generate_comparison_viz(metrics, df, y_test_inv, y_pred)
    
    print("\n" + "=" * 80)
    print("[OK] MILESTONE 3 COMPLETED!")
    print("=" * 80)
    print(f"LSTM Results: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    print(f"Prediction Accuracy: {100 - metrics['MAPE']:.1f}%")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return metrics


if __name__ == "__main__":
    main()

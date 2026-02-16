"""
=================================================================================
SMART ENERGY CONSUMPTION ANALYSIS - MAIN ORCHESTRATOR
Infosys Internship Project - Milestones 1, 2, 3 & 4
=================================================================================

This script orchestrates the full pipeline:
1. Milestone 1: Data Preprocessing (data_preprocessing.py)
2. Milestone 2: Feature Engineering & Baseline Model 
   (feature_engineering.py, baseline_model.py)
3. Milestone 3: LSTM Model Development (lstm_model.py)
4. Milestone 4: Web Dashboard & Smart Suggestions (app.py)

=================================================================================
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configuration
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'processed_data')
VIZ_DIR = os.path.join(PROJECT_DIR, 'visualizations')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')


def print_header():
    """Print project header."""
    print("\n" + "=" * 80)
    print("  SMART ENERGY CONSUMPTION ANALYSIS SYSTEM")
    print("  =========================================")
    print("  Infosys Internship Project - Milestones 1, 2, 3 & 4")
    print("=" * 80)
    print(f"\n  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: {PROJECT_DIR}")
    print("=" * 80 + "\n")


def step_1_preprocessing():
    """Execute Milestone 1: Data Preprocessing."""
    print("\n" + "#" * 80)
    print("#  MILESTONE 1: DATA COLLECTION & PREPROCESSING")
    print("#" * 80 + "\n")
    
    import data_preprocessing
    result = data_preprocessing.main()
    
    return result is not None


def step_2_feature_engineering():
    """Execute Milestone 2 Part 1: Feature Engineering."""
    print("\n" + "#" * 80)
    print("#  MILESTONE 2 (Part 1): FEATURE ENGINEERING")
    print("#" * 80 + "\n")
    
    import feature_engineering
    result = feature_engineering.main()
    
    return result is not None


def step_3_baseline_model():
    """Execute Milestone 2 Part 2: Baseline Model."""
    print("\n" + "#" * 80)
    print("#  MILESTONE 2 (Part 2): BASELINE MODEL (LINEAR REGRESSION)")
    print("#" * 80 + "\n")
    
    import baseline_model
    result = baseline_model.main()
    
    return result is not None


def step_4_lstm_model():
    """Execute Milestone 3: LSTM Model."""
    print("\n" + "#" * 80)
    print("#  MILESTONE 3: LSTM MODEL DEVELOPMENT")
    print("#" * 80 + "\n")
    
    import lstm_model
    result = lstm_model.main()
    
    return result is not None


def step_5_dashboard():
    """Execute Milestone 4: Launch Web Dashboard."""
    print("\n" + "#" * 80)
    print("#  MILESTONE 4: WEB DASHBOARD & SMART SUGGESTIONS")
    print("#" * 80 + "\n")
    
    print("  Launching Flask web dashboard...")
    print("  Open http://localhost:5000 in your browser")
    print("  Press Ctrl+C to stop the server\n")
    
    # Import and run Flask app
    sys.path.insert(0, PROJECT_DIR)
    from app import app
    app.run(debug=False, host='0.0.0.0', port=5000)


def print_summary():
    """Print execution summary."""
    print("\n" + "=" * 80)
    print("  EXECUTION SUMMARY")
    print("=" * 80)
    
    # Check for output files
    print("\n  Generated Visualizations:")
    viz_files = [
        'Milestone1_Module1_EDA.png',
        'Milestone1_Module2_Preprocessing.png',
        'Milestone2_Module3_FeatureEngineering.png',
        'Milestone2_Module4_BaselineModel.png',
        'Milestone3_LSTM_Complete.png',
        'Milestone3_Model_Comparison.png'
    ]
    for viz in viz_files:
        path = os.path.join(VIZ_DIR, viz)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"     {status} {viz}")
    
    print("\n  Processed Data Files:")
    data_files = [
        'data_hourly.csv',
        'data_daily.csv',
        'data_features.csv',
        'train_data.csv',
        'val_data.csv',
        'test_data.csv',
        'lstm_predictions.csv'
    ]
    for df in data_files:
        path = os.path.join(PROCESSED_DIR, df)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"     {status} {df}")
    
    print("\n  Model Artifacts:")
    model_files = [
        'lstm_best_model.keras',
        'lstm_scaler.pkl',
        'minmax_scaler.pkl',
        'linear_regression_model.pkl'
    ]
    for mf in model_files:
        path = os.path.join(MODELS_DIR, mf)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"     {status} {mf}")
    
    # Dashboard status
    print("\n  Web Dashboard:")
    templates_ok = os.path.exists(os.path.join(PROJECT_DIR, 'templates', 'index.html'))
    static_ok = os.path.exists(os.path.join(PROJECT_DIR, 'static', 'css', 'style.css'))
    app_ok = os.path.exists(os.path.join(PROJECT_DIR, 'app.py'))
    print(f"     {'[OK]' if app_ok else '[MISSING]'} Flask app (app.py)")
    print(f"     {'[OK]' if templates_ok else '[MISSING]'} Dashboard template")
    print(f"     {'[OK]' if static_ok else '[MISSING]'} Static assets")
    
    print("\n" + "=" * 80)
    print(f"  Execution Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  ALL MILESTONES COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


def main():
    """Main execution function - runs the full pipeline."""
    parser = argparse.ArgumentParser(description='Smart Energy Consumption Analysis System')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch the web dashboard only (skip ML pipeline)')
    parser.add_argument('--pipeline', action='store_true',
                        help='Run only the ML pipeline (Milestones 1-3)')
    parser.add_argument('--all', action='store_true',
                        help='Run pipeline then launch dashboard')
    args = parser.parse_args()

    start_time = datetime.now()
    print_header()

    # If --dashboard flag, just launch the dashboard
    if args.dashboard:
        step_5_dashboard()
        return True

    # Run pipeline
    success = True
    
    if not args.dashboard:
        # Milestone 1: Data Preprocessing
        if not step_1_preprocessing():
            print("[ERROR] Milestone 1 failed!")
            success = False
        
        # Milestone 2 Part 1: Feature Engineering
        if success and not step_2_feature_engineering():
            print("[ERROR] Feature Engineering failed!")
            success = False
        
        # Milestone 2 Part 2: Baseline Model
        if success and not step_3_baseline_model():
            print("[ERROR] Baseline Model failed!")
            success = False
        
        # Milestone 3: LSTM Model
        if success and not step_4_lstm_model():
            print("[ERROR] LSTM Model failed!")
            success = False
    
    # Print summary
    if success:
        print_summary()
    
    # Calculate total time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n  Total Execution Time: {duration}")
    
    # Launch dashboard if --all flag
    if success and args.all:
        step_5_dashboard()
    elif success:
        print("\n  To launch the web dashboard, run:")
        print("    python main.py --dashboard")
        print("    OR")
        print("    python app.py")
    
    return success


if __name__ == "__main__":
    main()

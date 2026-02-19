import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_static_plots(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(data_path)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    
    # 1. Hourly Consumption Trend (Last 7 days)
    plt.figure(figsize=(12, 6))
    df['Global_active_power'].iloc[-168:].plot(title='Energy Consumption Trend (Last 7 Days)')
    plt.xlabel('Timestamp')
    plt.ylabel('Active Power (kW)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'consumption_trend.png'))
    plt.close()
    
    # 2. Device-wise Usage (Sub-metering comparison)
    plt.figure(figsize=(10, 6))
    sub_metering = df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
    sub_metering.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.title('Average Device-Level Power Consumption')
    plt.ylabel('Energy (Wh)')
    plt.savefig(os.path.join(output_dir, 'device_insights.png'))
    plt.close()
    
    # 3. Peak vs Non-Peak Hourly Usage
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='IsPeak', y='Global_active_power', data=df)
    plt.title('Peak vs Non-Peak Hour Consumption')
    plt.savefig(os.path.join(output_dir, 'peak_comparison.png'))
    plt.close()
    
    print(f"Static visualizations generated in {output_dir}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'milestone2', 'feature_engineered_data.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'plots')
    generate_static_plots(DATA_PATH, OUTPUT_DIR)

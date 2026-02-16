"""
=================================================================================
SMART ENERGY CONSUMPTION ANALYSIS - SMART SUGGESTIONS ENGINE
Infosys Internship Project - Milestone 4
=================================================================================

Analyzes consumption patterns and generates:
- Device-specific energy saving tips
- Time-based optimization suggestions
- Anomaly detection alerts
- Cost estimates and savings potential
=================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime


# Average electricity rate (INR per kWh) - Indian residential tariff
ELECTRICITY_RATE = 7.0  # ₹/kWh (adjustable)
CURRENCY_SYMBOL = "₹"

DEVICE_MAPPING = {
    'Sub_metering_1': {
        'name': 'Kitchen',
        'appliances': 'Dishwasher, Microwave, Oven',
        'icon': '🍳',
        'color': '#e74c3c'
    },
    'Sub_metering_2': {
        'name': 'Laundry',
        'appliances': 'Washing Machine, Dryer, Refrigerator',
        'icon': '👕',
        'color': '#3498db'
    },
    'Sub_metering_3': {
        'name': 'HVAC',
        'appliances': 'Water Heater, Air Conditioning',
        'icon': '❄️',
        'color': '#2ecc71'
    }
}


def analyze_device_consumption(df):
    """Analyze device-level consumption patterns."""
    device_stats = {}
    device_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    total_consumption = sum(df[col].sum() for col in device_cols if col in df.columns)

    for col in device_cols:
        if col not in df.columns:
            continue
        device_info = DEVICE_MAPPING[col]
        values = df[col]

        stats = {
            'name': device_info['name'],
            'appliances': device_info['appliances'],
            'icon': device_info['icon'],
            'color': device_info['color'],
            'mean': round(float(values.mean()), 4),
            'max': round(float(values.max()), 4),
            'min': round(float(values.min()), 4),
            'std': round(float(values.std()), 4),
            'total': round(float(values.sum()), 2),
            'share_pct': round(float(values.sum() / total_consumption * 100), 1) if total_consumption > 0 else 0,
        }

        # Peak hours analysis
        if hasattr(df.index, 'hour'):
            hourly_avg = values.groupby(df.index.hour).mean()
            stats['peak_hour'] = int(hourly_avg.idxmax())
            stats['off_peak_hour'] = int(hourly_avg.idxmin())
            stats['peak_avg'] = round(float(hourly_avg.max()), 4)
            stats['off_peak_avg'] = round(float(hourly_avg.min()), 4)

        # Weekend vs weekday
        if hasattr(df.index, 'dayofweek'):
            weekday_avg = values[df.index.dayofweek < 5].mean()
            weekend_avg = values[df.index.dayofweek >= 5].mean()
            stats['weekday_avg'] = round(float(weekday_avg), 4)
            stats['weekend_avg'] = round(float(weekend_avg), 4)
            stats['weekend_increase'] = round(
                float((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg > 0 else 0, 1
            )

        device_stats[col] = stats

    return device_stats


def detect_anomalies(df, target_col='Global_active_power', threshold=2.5):
    """Detect anomalous consumption using z-score method."""
    if target_col not in df.columns:
        return []

    values = df[target_col]
    mean_val = values.mean()
    std_val = values.std()

    if std_val == 0:
        return []

    z_scores = (values - mean_val) / std_val
    anomalies = df[abs(z_scores) > threshold].copy()

    alerts = []
    for idx, row in anomalies.iterrows():
        z = float((row[target_col] - mean_val) / std_val)
        alerts.append({
            'timestamp': idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx),
            'value': round(float(row[target_col]), 4),
            'z_score': round(z, 2),
            'severity': 'HIGH' if abs(z) > 3.5 else 'MEDIUM',
            'type': 'spike' if z > 0 else 'drop',
            'message': (
                f"{'⚠️ High' if abs(z) > 3.5 else '⚡ Moderate'} consumption "
                f"{'spike' if z > 0 else 'drop'}: "
                f"{round(float(row[target_col]), 4)} kW "
                f"(z-score: {round(z, 2)})"
            )
        })

    # Limit to most recent/significant 20
    alerts.sort(key=lambda x: abs(x['z_score']), reverse=True)
    return alerts[:20]


def estimate_costs(df, rate=ELECTRICITY_RATE):
    """Estimate energy costs and savings potential."""
    if 'Global_active_power' not in df.columns:
        return {}

    # Calculate total consumption in kWh
    hours = len(df)  # each row = 1 hour for hourly data
    total_kwh = float(df['Global_active_power'].sum())

    # Monthly projection (assume ~730 hours/month)
    days_in_data = hours / 24 if hours > 0 else 1
    monthly_kwh = total_kwh / days_in_data * 30

    # Device-wise costs
    device_costs = {}
    for col, info in DEVICE_MAPPING.items():
        if col in df.columns:
            device_kwh = float(df[col].sum()) / 1000  # Convert Wh to kWh
            monthly_device_kwh = device_kwh / days_in_data * 30
            device_costs[info['name']] = {
                'total_kwh': round(device_kwh, 2),
                'monthly_kwh': round(monthly_device_kwh, 2),
                'monthly_cost': round(monthly_device_kwh * rate, 2),
                'icon': info['icon'],
                'color': info['color']
            }

    return {
        'total_kwh': round(total_kwh, 2),
        'monthly_kwh': round(monthly_kwh, 2),
        'monthly_cost': round(monthly_kwh * rate, 2),
        'daily_cost': round(monthly_kwh * rate / 30, 2),
        'annual_cost': round(monthly_kwh * rate * 12, 2),
        'rate': rate,
        'currency': CURRENCY_SYMBOL,
        'device_costs': device_costs,
        'data_hours': hours,
        'data_days': round(days_in_data, 1)
    }


def generate_suggestions(df, device_stats=None, anomalies=None, costs=None):
    """Generate smart energy-saving suggestions based on analysis."""
    if device_stats is None:
        device_stats = analyze_device_consumption(df)
    if anomalies is None:
        anomalies = detect_anomalies(df)
    if costs is None:
        costs = estimate_costs(df)

    suggestions = []

    # === Device-Specific Suggestions ===
    # Find highest consuming device
    device_shares = {k: v['share_pct'] for k, v in device_stats.items()}
    if device_shares:
        top_device_key = max(device_shares, key=device_shares.get)
        top_device = device_stats[top_device_key]

        suggestions.append({
            'category': 'Device Optimization',
            'priority': 'HIGH',
            'icon': '🔌',
            'title': f"{top_device['name']} is Your Biggest Consumer",
            'description': (
                f"{top_device['name']} ({top_device['appliances']}) accounts for "
                f"{top_device['share_pct']}% of total energy consumption. "
                f"Consider upgrading to energy-efficient appliances."
            ),
            'savings_potential': f"Up to 15-25% reduction possible",
            'color': top_device['color']
        })

    # HVAC specific
    hvac = device_stats.get('Sub_metering_3')
    if hvac and hvac['share_pct'] > 35:
        suggestions.append({
            'category': 'HVAC Optimization',
            'priority': 'HIGH',
            'icon': '🌡️',
            'title': 'Optimize HVAC Usage',
            'description': (
                f"HVAC consumes {hvac['share_pct']}% of energy. "
                "Set thermostat to 24°C in summer and 20°C in winter. "
                "Use programmable timers to reduce idle consumption."
            ),
            'savings_potential': "20-30% HVAC cost reduction",
            'color': '#2ecc71'
        })

    # Kitchen specific
    kitchen = device_stats.get('Sub_metering_1')
    if kitchen:
        suggestions.append({
            'category': 'Kitchen Efficiency',
            'priority': 'MEDIUM',
            'icon': '🍳',
            'title': 'Kitchen Energy Tips',
            'description': (
                "Use microwave instead of oven for reheating (saves 80% energy). "
                "Run dishwasher only with full loads. "
                f"Peak kitchen usage at {kitchen.get('peak_hour', 'N/A')}:00 — "
                "batch cook to reduce appliance cycling."
            ),
            'savings_potential': "10-15% kitchen energy savings",
            'color': '#e74c3c'
        })

    # Laundry specific
    laundry = device_stats.get('Sub_metering_2')
    if laundry:
        weekend_inc = laundry.get('weekend_increase', 0)
        suggestions.append({
            'category': 'Laundry Optimization',
            'priority': 'MEDIUM',
            'icon': '👕',
            'title': 'Optimize Laundry Schedule',
            'description': (
                f"Laundry consumption is {abs(weekend_inc):.0f}% "
                f"{'higher' if weekend_inc > 0 else 'lower'} on weekends. "
                "Wash with cold water (saves 90% of washing energy). "
                "Air-dry clothes when possible instead of using the dryer."
            ),
            'savings_potential': "15-20% laundry savings",
            'color': '#3498db'
        })

    # === Time-Based Suggestions ===
    if hasattr(df.index, 'hour') and 'Global_active_power' in df.columns:
        hourly_avg = df['Global_active_power'].groupby(df.index.hour).mean()
        peak_hour = int(hourly_avg.idxmax())
        off_peak_hour = int(hourly_avg.idxmin())

        suggestions.append({
            'category': 'Time-of-Use',
            'priority': 'HIGH',
            'icon': '⏰',
            'title': 'Shift Usage to Off-Peak Hours',
            'description': (
                f"Peak consumption occurs at {peak_hour}:00. "
                f"Lowest consumption at {off_peak_hour}:00. "
                "Schedule heavy appliances (washer, dryer, dishwasher) "
                "during off-peak hours (10 PM – 6 AM) to save on time-of-use tariffs."
            ),
            'savings_potential': "10-20% bill reduction with TOU tariff",
            'color': '#f39c12'
        })

    # === Anomaly-Based Suggestions ===
    high_anomalies = [a for a in anomalies if a['severity'] == 'HIGH']
    if high_anomalies:
        suggestions.append({
            'category': 'Anomaly Alert',
            'priority': 'HIGH',
            'icon': '🚨',
            'title': f'{len(high_anomalies)} Unusual Consumption Spikes Detected',
            'description': (
                "Multiple high-severity anomalies detected in your consumption data. "
                "This may indicate faulty appliances, phantom loads, or equipment "
                "left running unintentionally. Inspect devices during these periods."
            ),
            'savings_potential': "Preventing spikes can save 5-10%",
            'color': '#e74c3c'
        })

    # === Cost-Based Suggestions ===
    if costs:
        monthly_cost = costs.get('monthly_cost', 0)
        suggestions.append({
            'category': 'Cost Analysis',
            'priority': 'MEDIUM',
            'icon': '💰',
            'title': f'Monthly Bill Estimate: {CURRENCY_SYMBOL}{monthly_cost:,.0f}',
            'description': (
                f"Estimated monthly consumption: {costs.get('monthly_kwh', 0):.1f} kWh. "
                f"Annual projection: {CURRENCY_SYMBOL}{costs.get('annual_cost', 0):,.0f}. "
                "Implementing all suggestions could reduce costs by 20-35%."
            ),
            'savings_potential': f"{CURRENCY_SYMBOL}{monthly_cost * 0.25:,.0f}/month potential savings",
            'color': '#9b59b6'
        })

    # === General Tips ===
    suggestions.append({
        'category': 'General Tips',
        'priority': 'LOW',
        'icon': '💡',
        'title': 'Smart Energy Habits',
        'description': (
            "• Use LED bulbs (save 75% vs incandescent)\n"
            "• Unplug electronics when not in use (phantom loads = 5-10% of bill)\n"
            "• Use smart power strips for entertainment systems\n"
            "• Regular HVAC maintenance improves efficiency by 15%\n"
            "• Install motion sensors for outdoor and hallway lights"
        ),
        'savings_potential': "5-10% overall reduction",
        'color': '#f1c40f'
    })

    suggestions.append({
        'category': 'Renewable Energy',
        'priority': 'LOW',
        'icon': '☀️',
        'title': 'Consider Solar Installation',
        'description': (
            f"With {costs.get('monthly_kwh', 0):.0f} kWh monthly usage, "
            "a 3-5 kW rooftop solar system could offset 60-80% of consumption. "
            "Government subsidies available under PM-Surya Ghar scheme."
        ),
        'savings_potential': "60-80% bill reduction with solar",
        'color': '#e67e22'
    })

    return suggestions


def get_consumption_summary(df):
    """Get overall consumption summary statistics."""
    if 'Global_active_power' not in df.columns:
        return {}

    total_power = df['Global_active_power']

    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df.index.min().strftime('%Y-%m-%d') if hasattr(df.index.min(), 'strftime') else str(df.index.min()),
            'end': df.index.max().strftime('%Y-%m-%d') if hasattr(df.index.max(), 'strftime') else str(df.index.max()),
        },
        'global_power': {
            'mean': round(float(total_power.mean()), 4),
            'max': round(float(total_power.max()), 4),
            'min': round(float(total_power.min()), 4),
            'std': round(float(total_power.std()), 4),
            'median': round(float(total_power.median()), 4),
        },
        'devices_tracked': 3,
        'features_engineered': 53,
    }

    return summary


def get_full_analysis(df):
    """Run full analysis and return all results."""
    device_stats = analyze_device_consumption(df)
    anomalies = detect_anomalies(df)
    costs = estimate_costs(df)
    suggestions = generate_suggestions(df, device_stats, anomalies, costs)
    summary = get_consumption_summary(df)

    return {
        'summary': summary,
        'device_stats': device_stats,
        'anomalies': anomalies,
        'costs': costs,
        'suggestions': suggestions,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


if __name__ == '__main__':
    import os
    PROCESSED_DIR = r'c:\Suraj\My_files\Career\Infosys Spring Board\Project\processed_data'
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'data_hourly.csv'), index_col=0, parse_dates=True)
    results = get_full_analysis(df)

    print("\n=== SMART SUGGESTIONS ===")
    for s in results['suggestions']:
        print(f"\n{s['icon']} [{s['priority']}] {s['title']}")
        print(f"   {s['description']}")
        print(f"   💰 {s['savings_potential']}")

    print(f"\n=== COST ESTIMATE ===")
    print(f"   Monthly: {results['costs']['currency']}{results['costs']['monthly_cost']:,.2f}")
    print(f"   Annual:  {results['costs']['currency']}{results['costs']['annual_cost']:,.2f}")

    print(f"\n=== ANOMALIES ({len(results['anomalies'])}) ===")
    for a in results['anomalies'][:5]:
        print(f"   {a['message']}")

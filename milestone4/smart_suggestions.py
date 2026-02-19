import pandas as pd
import numpy as np

def generate_suggestions(stats):
    """
    Generate energy-saving suggestions based on consumption statistics.
    stats: dict containing 'avg_power', 'peak_hours', 'high_metering'
    """
    suggestions = []
    
    avg_power = stats.get('avg_power', 0)
    if avg_power > 2.0:
        suggestions.append({
            "title": "Reduce Base Load",
            "description": "Your average active power is higher than usual. Check for devices left on standby or inefficient appliances.",
            "impact": "High"
        })
    else:
        suggestions.append({
            "title": "Good Energy Habits",
            "description": "Your average energy usage is within the efficient range. Keep it up!",
            "impact": "Neutral"
        })

    peak_count = stats.get('peak_hour_count', 0)
    if peak_count > 5:
        suggestions.append({
            "title": "Shift Peak Usage",
            "description": f"You had {peak_count} peak hours yesterday. Try shifting high-energy tasks like laundry to non-peak hours.",
            "impact": "Medium"
        })

    metering_3 = stats.get('sub_metering_3_avg', 0)
    if metering_3 > 10:
        suggestions.append({
            "title": "Check HVAC/Water Heater",
            "description": "High consumption detected in Sub-metering 3. Ensure your climate control systems are set to optimal temperatures.",
            "impact": "Medium"
        })
        
    if stats.get('weekend_spike', False):
        suggestions.append({
            "title": "Weekend Optimization",
            "description": "We noticed a spike in your weekend usage. Consider scheduling energy-intensive chores more efficiently.",
            "impact": "Low"
        })

    return suggestions

def get_consumption_stats(df):
    """
    Extract relevant statistics from the dataframe for suggestion generation.
    """
    last_24h = df.iloc[-24:]
    stats = {
        "avg_power": last_24h['Global_active_power'].mean(),
        "peak_hour_count": (last_24h['IsPeak'] == 1).sum(),
        "sub_metering_3_avg": last_24h['Sub_metering_3'].mean(),
        "weekend_spike": df.iloc[-48:]['IsWeekend'].any() and df.iloc[-48:]['Global_active_power'].max() > df['Global_active_power'].mean() * 1.5
    }
    return stats

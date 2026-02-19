import pandas as pd
from config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Home ID": "home_id",
        "Appliance Type": "appliance",
        "Energy Consumption (kWh)": "energy",
        "Outdoor Temperature (°C)": "temperature",
        "Household Size": "household_size"
    })

    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour


    return df


def device_summary(df):
    summary = df.groupby("appliance")["energy"].mean().reset_index()
    return summary.to_dict(orient="records")


def time_series_data(df):
    ts = df.groupby("timestamp")["energy"].sum().reset_index()
    return ts.to_dict(orient="records")

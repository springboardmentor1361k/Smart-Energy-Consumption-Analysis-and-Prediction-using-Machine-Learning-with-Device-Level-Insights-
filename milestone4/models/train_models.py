import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from config import *

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df = df.rename(columns={
    "Appliance Type": "appliance",
    "Energy Consumption (kWh)": "energy",
    "Outdoor Temperature (°C)": "temperature",
    "Household Size": "household_size",
    "Season": "season"
})

# Extract hour from Time
df["hour"] = pd.to_datetime(df["Time"]).dt.hour

# ==============================
# ENCODING
# ==============================

# Encode Appliance
le = LabelEncoder()
df["appliance_encoded"] = le.fit_transform(df["appliance"])

# Encode Season
season_map = {
    "Winter": 0,
    "Spring": 1,
    "Summer": 2,
    "Fall": 3
}
df["season_encoded"] = df["season"].map(season_map)

# ==============================
# LINEAR REGRESSION MODEL
# ==============================

X = df[["appliance_encoded", "hour", "temperature", "household_size", "season_encoded"]]
y = df["energy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate
lr_preds = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_preds)

print("Linear Regression MSE:", round(lr_mse, 4))

# Save Linear model and encoder
joblib.dump(lr, LINEAR_MODEL_PATH)
joblib.dump(le, ENCODER_PATH)

# ==============================
# LSTM MODEL (Time Series)
# ==============================

scaler = MinMaxScaler()
scaled_energy = scaler.fit_transform(df[["energy"]])

window = 24  # 24 hours window
X_lstm, y_lstm = [], []

for i in range(window, len(scaled_energy)):
    X_lstm.append(scaled_energy[i-window:i])
    y_lstm.append(scaled_energy[i])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# Train/Test split for LSTM
split = int(0.8 * len(X_lstm))
X_lstm_train, X_lstm_test = X_lstm[:split], X_lstm[split:]
y_lstm_train, y_lstm_test = y_lstm[:split], y_lstm[split:]

# Build Model
model = Sequential()
model.add(LSTM(64, input_shape=(window, 1)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(
    X_lstm_train,
    y_lstm_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_lstm_test, y_lstm_test),
    verbose=1
)

# Evaluate LSTM
lstm_preds = model.predict(X_lstm_test)
lstm_preds = scaler.inverse_transform(lstm_preds)
y_lstm_test_inv = scaler.inverse_transform(y_lstm_test)

lstm_mse = mean_squared_error(y_lstm_test_inv, lstm_preds)
print("LSTM MSE:", round(lstm_mse, 4))

# Save models
model.save(LSTM_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n Models trained and saved successfully.")

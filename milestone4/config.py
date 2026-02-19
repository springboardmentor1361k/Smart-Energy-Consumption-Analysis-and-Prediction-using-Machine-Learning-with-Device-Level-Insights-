import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "smart_home_energy_consumption_large.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")

LINEAR_MODEL_PATH = os.path.join(MODEL_DIR, "linear_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
EPOCHS = 50
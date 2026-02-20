from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained LSTM model
model = load_model("models/final_lstm_model.keras")
print("Model loaded successfully!")

# Fixed values (from your notebook mean values)
LAG_1 = 1.113806
LAG_24 = 1.114990
ROLLING_MEAN = 1.114255
ROLLING_STD = 0.780879


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        hour = float(data["hour"])
        day = float(data["day"])
        month = float(data["month"])

        # Prepare full feature vector (8 features as trained)
        features = np.array([[
            hour,
            day,
            month,
            LAG_1,
            LAG_24,
            ROLLING_MEAN,
            ROLLING_STD,
            0  # dummy feature if your model expects 8 inputs
        ]], dtype=np.float32)

        # Reshape for LSTM (samples, timesteps, features)
        features = features.reshape((1, 1, 8))

        prediction = model.predict(features)[0][0]

        print("Raw prediction:", prediction)

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

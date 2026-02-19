from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load trained model
try:
    model = load_model("energy_lstm_model.h5", compile=False)
except Exception as e:
    print("Model load error:", e)
    model = None

SEQ_LEN = 7


@app.route("/")
def home():
    return "Energy Prediction Backend Running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.json
        values = data.get('values')

        if not values or len(values) != SEQ_LEN:
            return jsonify({
                "error": f"Provide exactly {SEQ_LEN} values"
            }), 400

        values = np.array(values).reshape(1, SEQ_LEN, 1)

        prediction = model.predict(values)[0][0]

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import numpy as np
import random
from datetime import datetime

app = Flask(__name__)

# =====================================================
# 🔹 Energy Analytics Engine
# =====================================================

class EnergyEngine:

    @staticmethod
    def estimate_next_cycle(readings):
        """
        Simulated forecasting logic.
        Uses weighted mean + small random variance.
        """
        readings = np.array(readings)
        weights = np.linspace(1, 2, len(readings))
        weighted_avg = np.average(readings, weights=weights)

        noise = random.uniform(-0.3, 0.3)
        return round(float(weighted_avg + noise), 2)

    @staticmethod
    def build_advisory(predicted_value):
        """
        Generates optimization recommendations.
        """
        advisory = []

        if predicted_value >= 7:
            advisory.extend([
                "Peak load warning — consider shifting heavy appliances.",
                "Activate energy saver mode for HVAC systems.",
                "Evaluate standby power losses."
            ])

        elif 4 <= predicted_value < 7:
            advisory.extend([
                "Stable consumption zone.",
                "Review mid-load appliances for efficiency gain.",
                "Schedule high-energy tasks during off-peak hours."
            ])

        else:
            advisory.extend([
                "Efficient energy behavior detected.",
                "System operating within optimal threshold."
            ])

        return advisory


# =====================================================
# 🔹 Routes
# =====================================================

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_energy():
    try:
        payload = request.get_json(force=True)

        readings = payload.get("values", [])

        if not isinstance(readings, list) or len(readings) == 0:
            return jsonify({
                "status": "failed",
                "message": "Energy readings are required."
            }), 400

        readings = [float(v) for v in readings]

        # 🔮 Forecast
        forecast = EnergyEngine.estimate_next_cycle(readings)

        # 💡 Advisory
        recommendations = EnergyEngine.build_advisory(forecast)

        response = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "forecast_kw": forecast,
            "insights": recommendations,
            "input_count": len(readings)
        }

        return jsonify(response)

    except Exception as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


# =====================================================
# 🔹 Application Entry
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
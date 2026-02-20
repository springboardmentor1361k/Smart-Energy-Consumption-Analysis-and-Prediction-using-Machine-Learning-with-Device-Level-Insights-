from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None


app = Flask(__name__)


@dataclass
class InferenceArtifacts:
    model: Any | None
    scaler: Any | None


def _load_artifacts() -> InferenceArtifacts:
    model = None
    scaler = None
    try:
        scaler = joblib.load("energy_scaler.pkl")
    except Exception:
        scaler = None

    if load_model is not None:
        try:
            model = load_model("lstm_energy_model.keras")
        except Exception:
            try:
                model = load_model("lstm_energy_model.h5")
            except Exception:
                model = None
    return InferenceArtifacts(model=model, scaler=scaler)


ARTIFACTS = _load_artifacts()


def _base_dashboard_data() -> dict[str, Any]:
    daily_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_consumption = [17.1, 16.4, 18.8, 19.2, 20.1, 14.8, 15.5]

    device_labels = [
        "HVAC",
        "Refrigerator",
        "Washing Machine",
        "Lighting",
        "Water Heater",
    ]
    device_consumption = [41, 13, 11, 19, 16]

    appliance_labels = ["Cooling", "Kitchen", "Laundry", "Lighting", "Other"]
    appliance_percentages = [38, 21, 13, 18, 10]

    total = round(float(np.sum(daily_consumption)), 1)
    predicted_next = round(float(np.mean(daily_consumption) * 1.05), 1)
    cost_estimation = round(total * 0.18, 2)

    return {
        "cards": {
            "total_consumption": total,
            "predicted_next_day": predicted_next,
            "peak_usage_hours": "6 PM - 10 PM",
            "cost_estimation": cost_estimation,
            "sustainability_score": 82,
        },
        "charts": {
            "daily": {"labels": daily_labels, "values": daily_consumption},
            "device": {"labels": device_labels, "values": device_consumption},
            "appliance": {"labels": appliance_labels, "values": appliance_percentages},
        },
    }


def _build_insights_data(dashboard_data: dict[str, Any]) -> dict[str, Any]:
    daily = dashboard_data["charts"]["daily"]["values"]
    max_value = max(daily)
    min_value = min(daily)
    peak_day = dashboard_data["charts"]["daily"]["labels"][daily.index(max_value)]
    low_day = dashboard_data["charts"]["daily"]["labels"][daily.index(min_value)]
    avg_daily = round(float(np.mean(daily)), 2)

    return {
        "peak_day": peak_day,
        "peak_value": max_value,
        "lowest_day": low_day,
        "lowest_value": min_value,
        "average_daily": avg_daily,
        "insights": [
            "Peak demand appears in the evening, especially between 6 PM and 10 PM.",
            "Cooling and kitchen appliances drive most of the total load share.",
            "Weekend usage is lower, indicating flexible loads can be shifted.",
        ],
    }


def _build_reports_data(dashboard_data: dict[str, Any]) -> dict[str, Any]:
    total = dashboard_data["cards"]["total_consumption"]
    predicted_next = dashboard_data["cards"]["predicted_next_day"]
    cost = dashboard_data["cards"]["cost_estimation"]
    efficiency = dashboard_data["cards"]["sustainability_score"]

    return {
        "monthly_projection_kwh": round(predicted_next * 30, 2),
        "monthly_projection_cost": round(float(cost) * 4.1, 2),
        "weekly_summary": {
            "total_kwh": total,
            "avg_kwh_per_day": round(total / 7, 2),
            "efficiency_score": efficiency,
        },
        "report_rows": [
            ["Week 1", 121.4, 22.1, "Stable"],
            ["Week 2", 117.9, 21.2, "Improved"],
            ["Week 3", 123.3, 22.8, "Peak Weather Load"],
            ["Week 4", 115.7, 20.6, "Optimized"],
        ],
    }


def _rule_based_prediction(features: dict[str, float]) -> float:
    temp = features["temperature"]
    humidity = features["humidity"]
    voltage = features["voltage"]
    current = features["current"]
    usage = features["device_usage"]

    return (
        (0.25 * temp)
        + (0.12 * humidity)
        + (0.08 * max(voltage - 210, 0))
        + (2.2 * current)
        + (2.8 * usage)
    )


def _model_prediction(features: dict[str, float]) -> float:
    values = np.array(
        [
            features["temperature"],
            features["humidity"],
            features["voltage"],
            features["current"],
            features["device_usage"],
        ],
        dtype=float,
    ).reshape(1, -1)

    model = ARTIFACTS.model
    scaler = ARTIFACTS.scaler
    if model is None or scaler is None:
        return _rule_based_prediction(features)

    try:
        scaled = scaler.transform(values)
        # LSTM expects [batch, timesteps, features]. We use one timestep here.
        sequence = scaled.reshape(1, 1, scaled.shape[1])
        prediction = model.predict(sequence, verbose=0)
        return float(np.squeeze(prediction))
    except Exception:
        return _rule_based_prediction(features)


def _recommendations(predicted_kwh: float, features: dict[str, float]) -> list[str]:
    tips: list[str] = []

    # Base suggestions from predicted consumption level.
    if predicted_kwh > 22:
        tips.extend(
            [
                "Reduce usage during peak hours (6 PM - 10 PM).",
                "Shift heavy loads to off-peak windows after 10 PM.",
                "Set HVAC 1-2 C higher to reduce compressor runtime.",
                "Prioritize inverter-based appliances for long operating cycles.",
            ]
        )
    elif predicted_kwh > 15:
        tips.extend(
            [
                "Schedule laundry and water heater cycles in off-peak periods.",
                "Use eco mode for high-load appliances where available.",
                "Group similar appliance tasks to reduce repeated startup losses.",
            ]
        )
    else:
        tips.extend(
            [
                "Great efficiency today. Keep current usage habits.",
                "Maintain current thermostat schedule and avoid unnecessary runtime.",
            ]
        )

    # Smart, feature-aware tips.
    if features["temperature"] >= 32:
        tips.append("Use blinds and pre-cooling to lower afternoon HVAC demand.")
    if features["humidity"] >= 65:
        tips.append("Run dehumidification in short cycles to reduce AC workload.")
    if features["voltage"] >= 235:
        tips.append("Use surge-protected smart strips and avoid unnecessary standby loads.")
    if features["current"] >= 8:
        tips.append("Avoid running multiple high-current devices simultaneously.")
    if features["device_usage"] >= 8:
        tips.append("Set auto-off timers for long-running appliances.")
    if features["device_usage"] <= 4:
        tips.append("You have spare capacity; keep flexible loads in off-peak slots.")

    # Always include one universal recommendation.
    tips.append("Track weekly usage trends and target a 5-10% reduction month-over-month.")
    tips.append("Clean AC filters and refrigerator coils monthly to improve efficiency.")
    tips.append("Replace old bulbs with LEDs and use occupancy-based lighting controls.")

    # De-duplicate while preserving order, then cap list size.
    deduped: list[str] = []
    for tip in tips:
        if tip not in deduped:
            deduped.append(tip)
    return deduped[:8]


def _risk_level(predicted_kwh: float) -> str:
    if predicted_kwh >= 22:
        return "High"
    if predicted_kwh >= 15:
        return "Moderate"
    return "Low"


@app.route("/")
def index():
    dashboard_data = _base_dashboard_data()
    return render_template(
        "index.html",
        dashboard_data=dashboard_data,
        active_page="dashboard",
    )


@app.route("/predictions")
def predictions_page():
    dashboard_data = _base_dashboard_data()
    return render_template(
        "predictions.html",
        dashboard_data=dashboard_data,
        active_page="predictions",
    )


@app.route("/insights")
def insights_page():
    dashboard_data = _base_dashboard_data()
    insights_data = _build_insights_data(dashboard_data)
    return render_template(
        "insights.html",
        dashboard_data=dashboard_data,
        insights_data=insights_data,
        active_page="insights",
    )


@app.route("/device-analytics")
def device_analytics_page():
    dashboard_data = _base_dashboard_data()
    return render_template(
        "device_analytics.html",
        dashboard_data=dashboard_data,
        active_page="device_analytics",
    )


@app.route("/reports")
def reports_page():
    dashboard_data = _base_dashboard_data()
    reports_data = _build_reports_data(dashboard_data)
    return render_template(
        "reports.html",
        dashboard_data=dashboard_data,
        reports_data=reports_data,
        active_page="reports",
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    required = ["temperature", "humidity", "voltage", "current", "device_usage"]
    missing = [k for k in required if k not in payload or str(payload[k]).strip() == ""]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        features = {k: float(payload[k]) for k in required}
    except (TypeError, ValueError):
        return jsonify({"error": "All inputs must be numeric."}), 400

    predicted = round(_model_prediction(features), 2)
    score = int(max(0, min(100, 100 - (predicted * 2.1))))
    estimated_cost_inr = round(predicted * 8.0, 2)
    risk_level = _risk_level(predicted)

    return jsonify(
        {
            "prediction_kwh": predicted,
            "risk_level": risk_level,
            "estimated_cost_inr": estimated_cost_inr,
            "recommendations": _recommendations(predicted, features),
            "sustainability_score": score,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

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

    daily_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    device_labels = [
        "HVAC",
        "Refrigerator",
        "Washing Machine",
        "Lighting",
        "Water Heater",
    ]

    week_defs = [
        {
            "name": "Week 1",
            "total": 121.4,
            "cost": 22.1,
            "status": "Stable",
            "efficiency_score": 79,
            "factors": [1.02, 0.98, 1.05, 1.08, 1.12, 0.85, 0.82],
            "device": [41, 13, 11, 19, 16],
        },
        {
            "name": "Week 2",
            "total": 117.9,
            "cost": 21.2,
            "status": "Improved",
            "efficiency_score": 86,
            "factors": [0.98, 0.96, 1.0, 1.02, 1.05, 0.88, 0.86],
            "device": [38, 14, 12, 18, 18],
        },
        {
            "name": "Week 3",
            "total": 123.3,
            "cost": 22.8,
            "status": "Peak Weather Load",
            "efficiency_score": 71,
            "factors": [1.08, 1.05, 1.12, 1.15, 1.18, 0.92, 0.88],
            "device": [45, 12, 10, 17, 16],
        },
        {
            "name": "Week 4",
            "total": 115.7,
            "cost": 20.6,
            "status": "Optimized",
            "efficiency_score": 89,
            "factors": [0.95, 0.93, 0.98, 1.0, 1.02, 0.84, 0.82],
            "device": [36, 15, 13, 19, 17],
        },
    ]

    weeks: dict[str, Any] = {}
    report_rows: list[list[Any]] = []

    for week in week_defs:
        factor_sum = float(np.sum(week["factors"]))
        daily_values = [
            round(week["total"] * factor / factor_sum, 1) for factor in week["factors"]
        ]
        weeks[week["name"]] = {
            "daily": {"labels": daily_labels, "values": daily_values},
            "device": {"labels": device_labels, "values": week["device"]},
            "status": week["status"],
            "total_kwh": week["total"],
            "cost": week["cost"],
            "efficiency_score": week["efficiency_score"],
        }
        report_rows.append([week["name"], week["total"], week["cost"], week["status"]])

    return {
        "monthly_projection_kwh": round(predicted_next * 30, 2),
        "monthly_projection_cost": round(float(cost) * 4.1, 2),
        "weekly_summary": {
            "total_kwh": total,
            "avg_kwh_per_day": round(total / 7, 2),
            "efficiency_score": efficiency,
        },
        "report_rows": report_rows,
        "weeks": weeks,
        "default_week": "Week 1",
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


def _build_analytics_from_prediction(
    predicted_kwh: float, features: dict[str, float]
) -> dict[str, Any]:
    daily_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_factors = [1.02, 0.98, 1.05, 1.08, 1.12, 0.85, 0.82]
    temp_weight = 1 + (features["temperature"] - 25) * 0.01
    usage_weight = 1 + (features["device_usage"] - 5) * 0.02

    daily_values = [
        round(predicted_kwh * factor * temp_weight * usage_weight, 1)
        for factor in weekday_factors
    ]
    total = round(float(np.sum(daily_values)), 1)

    if features["device_usage"] >= 8 or features["temperature"] >= 32:
        peak_hours = "6 PM - 10 PM"
    elif features["device_usage"] >= 5:
        peak_hours = "5 PM - 9 PM"
    else:
        peak_hours = "2 PM - 6 PM"

    device_labels = [
        "HVAC",
        "Refrigerator",
        "Washing Machine",
        "Lighting",
        "Water Heater",
    ]
    hvac = min(48.0, 18 + features["temperature"] * 0.9 + features["humidity"] * 0.1)
    water_heater = min(22.0, 8 + features["current"] * 1.2)
    laundry = min(18.0, 6 + features["device_usage"] * 1.1)
    lighting = min(20.0, 10 + max(features["voltage"] - 220, 0) * 0.15 + 8)
    refrigerator = max(8.0, 100 - hvac - water_heater - laundry - lighting)
    raw_shares = [hvac, refrigerator, laundry, lighting, water_heater]
    share_total = sum(raw_shares)
    device_values = [int(round(v / share_total * 100)) for v in raw_shares]

    appliance_labels = ["Cooling", "Kitchen", "Laundry", "Lighting", "Other"]
    cooling = int(round(device_values[0] * 0.95))
    kitchen = int(round(device_values[1] * 0.6 + device_values[3] * 0.2))
    laundry_pct = device_values[2]
    lighting_pct = int(round(device_values[3] * 0.7))
    other = max(5, 100 - cooling - kitchen - laundry_pct - lighting_pct)
    appliance_values = [cooling, kitchen, laundry_pct, lighting_pct, other]

    score = int(max(0, min(100, 100 - (predicted_kwh * 2.1))))
    cost = round(predicted_kwh * 8.0, 2)

    max_value = max(daily_values)
    min_value = min(daily_values)

    return {
        "cards": {
            "total_consumption": total,
            "predicted_next_day": round(predicted_kwh, 2),
            "peak_usage_hours": peak_hours,
            "cost_estimation": cost,
            "sustainability_score": score,
        },
        "charts": {
            "daily": {"labels": daily_labels, "values": daily_values},
            "device": {"labels": device_labels, "values": device_values},
            "appliance": {"labels": appliance_labels, "values": appliance_values},
        },
        "insights": {
            "peak_day": daily_labels[daily_values.index(max_value)],
            "peak_value": max_value,
            "lowest_day": daily_labels[daily_values.index(min_value)],
            "lowest_value": min_value,
            "average_daily": round(float(np.mean(daily_values)), 2),
        },
    }


def _empty_dashboard_data() -> dict[str, Any]:
    return {
        "cards": {},
        "charts": {
            "daily": {"labels": [], "values": []},
            "device": {"labels": [], "values": []},
            "appliance": {"labels": [], "values": []},
        },
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        dashboard_data=_empty_dashboard_data(),
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
    analytics = _build_analytics_from_prediction(predicted, features)

    return jsonify(
        {
            "prediction_kwh": predicted,
            "risk_level": risk_level,
            "estimated_cost_inr": estimated_cost_inr,
            "recommendations": _recommendations(predicted, features),
            "sustainability_score": score,
            "analytics": analytics,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

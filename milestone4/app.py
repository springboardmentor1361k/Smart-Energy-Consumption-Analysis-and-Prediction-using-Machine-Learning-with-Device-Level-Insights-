from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import importlib

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    keras_models = importlib.import_module("keras.models")
    load_model = getattr(keras_models, "load_model", None)
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

IDEAL_DAILY_KWH = 12.0
COST_PER_KWH_INR = 8.0
DAILY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_FACTORS = [1.02, 0.98, 1.05, 1.08, 1.12, 0.85, 0.82]
HOURLY_LABELS = ["6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM", "12 AM"]
DEVICE_LABELS = [
    "HVAC",
    "Refrigerator",
    "Washing Machine",
    "Lighting",
    "Water Heater",
]
APPLIANCE_LABELS = ["Cooling", "Kitchen", "Laundry", "Lighting", "Other"]

FORECAST_SCENARIOS = [
    {
        "name": "Current Prediction",
        "factor": 1.0,
        "status": "Baseline forecast",
        "description": "Projected from your latest dashboard prediction.",
    },
    {
        "name": "Higher Usage Scenario",
        "factor": 1.03,
        "status": "Projected +3% usage",
        "description": (
            "Simulated increase with higher runtime across HVAC, laundry, "
            "lighting, and water heating."
        ),
    },
    {
        "name": "Peak Demand Scenario",
        "factor": 1.06,
        "status": "Projected peak load",
        "description": (
            "Simulated peak period with the largest share shifts to HVAC "
            "and water heating."
        ),
    },
    {
        "name": "Optimized Efficiency Scenario",
        "factor": 0.94,
        "status": "Projected savings",
        "description": (
            "Simulated savings with reduced HVAC, lighting, laundry, "
            "and water heating use."
        ),
    },
]

# Device order: HVAC, Refrigerator, Washing Machine, Lighting, Water Heater
SCENARIO_DEVICE_WEIGHT_MULT = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.04, 1.0, 1.10, 1.08, 1.05],
    [1.15, 1.0, 1.06, 1.10, 1.18],
    [0.88, 1.0, 0.90, 0.82, 0.86],
]

SUSTAINABILITY_FORMULA = (
    "Score (0-100) = Consumption (40 pts) + Peak Load (25 pts) + "
    "Device Efficiency (25 pts) + Off-peak Usage (10 pts). "
    "Lower predicted kWh, lower peak-hour load, and higher device efficiency increase the score."
)
EFFICIENCY_FORMULA = (
    "Energy Efficiency = (Ideal Consumption / Actual Consumption) x 100, "
    f"where ideal baseline is {IDEAL_DAILY_KWH} kWh/day. Result is clamped between 0 and 100."
)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _normalize_percentages(values: list[float]) -> list[int]:
    total = float(sum(values))
    count = len(values)
    if count == 0:
        return []
    if total <= 0:
        base = 100 // count
        result = [base] * count
        result[0] += 100 - sum(result)
        return result

    scaled = [value / total * 100 for value in values]
    rounded = [int(round(value)) for value in scaled]
    diff = 100 - sum(rounded)
    if diff:
        adjust_index = max(range(count), key=lambda index: scaled[index])
        rounded[adjust_index] += diff
    return rounded


def _distribute_kwh(total: float, weights: list[float]) -> list[float]:
    weight_sum = float(sum(weights))
    count = len(weights)
    if count == 0:
        return []
    if weight_sum <= 0:
        equal = round(total / count, 2)
        values = [equal] * count
    else:
        values = [round(total * weight / weight_sum, 2) for weight in weights]

    diff = round(total - sum(values), 2)
    if diff:
        values[0] = round(values[0] + diff, 2)
    return values


def _device_weights(features: dict[str, float]) -> list[float]:
    temperature = features["temperature"]
    humidity = features["humidity"]
    voltage = features["voltage"]
    current = features["current"]
    usage = features["device_usage"]

    return [
        max(0.5, temperature * 0.85 + humidity * 0.08 + usage * 0.45),
        max(0.3, 2.0 + voltage * 0.008),
        max(0.2, usage * 0.55 + current * 0.15),
        max(0.2, usage * 0.35 + max(voltage - 220.0, 0.0) * 0.04),
        max(0.3, current * 1.1 + temperature * 0.05),
    ]


def _hourly_weights(features: dict[str, float]) -> list[float]:
    temperature = features["temperature"]
    current = features["current"]
    usage = features["device_usage"]

    morning = 0.9 + usage * 0.02
    afternoon = 1.0 + temperature * 0.015
    evening = 1.15 + usage * 0.04 + current * 0.01
    night = 0.75 + usage * 0.01

    return [
        morning,
        morning * 1.05,
        afternoon,
        afternoon * 0.95,
        evening,
        evening * 1.08,
        night,
    ]


def _device_efficiency_scores(
    features: dict[str, float], device_percentages: list[int]
) -> list[int]:
    temperature = features["temperature"]
    voltage = features["voltage"]
    current = features["current"]
    usage = features["device_usage"]

    scores = [
        _clamp(92 - (temperature - 22) * 2.2 - device_percentages[0] * 0.12, 40, 95),
        _clamp(88 + (235 - voltage) * 0.25, 50, 98),
        _clamp(80 - usage * 1.4, 45, 90),
        _clamp(75 + (6 - usage) * 2.0, 55, 95),
        _clamp(78 - current * 1.15, 40, 88),
    ]
    return [int(round(score)) for score in scores]


def _peak_window(labels: list[str], values: list[float]) -> dict[str, Any]:
    peak_index = values.index(max(values))
    peak_value = values[peak_index]
    start_index = max(0, peak_index - 1)
    end_index = min(len(labels) - 1, peak_index + 1)
    start = labels[start_index]
    end = labels[end_index]

    return {
        "start": start,
        "end": end,
        "peak_kwh": peak_value,
        "display": f"{start} - {end}",
        "detail": f"{start} - {end} ({peak_value} kWh peak)",
    }


def _energy_efficiency_score(predicted_kwh: float) -> int:
    if predicted_kwh <= 0:
        return 0
    return int(round(_clamp((IDEAL_DAILY_KWH / predicted_kwh) * 100)))


def _efficiency_rating(score: int) -> str:
    if score >= 86:
        return "Excellent"
    if score >= 66:
        return "Good"
    if score >= 41:
        return "Average"
    return "Poor"


def _sustainability_score(
    predicted_kwh: float,
    features: dict[str, float],
    peak_kwh: float,
    avg_device_efficiency: float,
) -> int:
    consumption_points = _clamp((IDEAL_DAILY_KWH / max(predicted_kwh, 0.1)) * 40, 0, 40)
    peak_ratio = peak_kwh / max(predicted_kwh, 0.1)
    peak_points = _clamp(25 * (1 - peak_ratio), 0, 25)
    efficiency_points = _clamp(avg_device_efficiency * 0.25, 0, 25)

    usage = features["device_usage"]
    if usage <= 5:
        renewable_points = 10.0
    elif usage <= 8:
        renewable_points = 5.0
    else:
        renewable_points = 0.0

    total = consumption_points + peak_points + efficiency_points + renewable_points
    return int(round(_clamp(total)))


def _build_insights_text(
    analytics: dict[str, Any], features: dict[str, float], predicted_kwh: float
) -> list[str]:
    peak = analytics["cards"]["peak_usage"]
    insights = analytics["insights"]
    appliance = analytics["charts"]["appliance"]
    top_index = appliance["values"].index(max(appliance["values"]))

    return [
        (
            f"Peak demand occurs between {peak['start']} and {peak['end']} "
            f"with {peak['peak_kwh']} kWh load."
        ),
        (
            f"{appliance['labels'][top_index]} drives the largest share at "
            f"{appliance['values'][top_index]}% of predicted daily use."
        ),
        (
            f"{insights['peak_day']} is the highest day at {insights['peak_value']} kWh; "
            f"{insights['lowest_day']} is lowest at {insights['lowest_value']} kWh."
        ),
        (
            f"Predicted next-day consumption is {predicted_kwh} kWh with "
            f"{analytics['cards']['efficiency_rating']} energy efficiency."
        ),
    ]


def _empty_dashboard_data() -> dict[str, Any]:
    return {
        "cards": {},
        "charts": {
            "daily": {"labels": [], "values": []},
            "device": {"labels": [], "values": [], "kwh": []},
            "appliance": {"labels": [], "values": [], "kwh": []},
            "hourly": {"labels": [], "values": []},
            "device_efficiency": {"labels": [], "values": []},
        },
        "insights": {},
    }


def _build_analytics_from_prediction(
    predicted_kwh: float, features: dict[str, float]
) -> dict[str, Any]:
    temp_factor = 1 + (features["temperature"] - 25) * 0.008
    usage_factor = 1 + (features["device_usage"] - 5) * 0.015
    daily_values = [
        round(predicted_kwh * factor * temp_factor * usage_factor, 1)
        for factor in WEEKDAY_FACTORS
    ]
    weekly_total = round(float(np.sum(daily_values)), 1)

    device_weights = _device_weights(features)
    device_kwh = _distribute_kwh(predicted_kwh, device_weights)
    device_percentages = _normalize_percentages(device_kwh)

    appliance_kwh = [
        device_kwh[0],
        device_kwh[1],
        device_kwh[2],
        device_kwh[3],
        device_kwh[4],
    ]
    appliance_percentages = _normalize_percentages(appliance_kwh)

    hourly_values = _distribute_kwh(predicted_kwh, _hourly_weights(features))
    peak = _peak_window(HOURLY_LABELS, hourly_values)

    device_efficiency = _device_efficiency_scores(features, device_percentages)
    avg_device_efficiency = float(np.mean(device_efficiency))
    sustainability = _sustainability_score(
        predicted_kwh, features, peak["peak_kwh"], avg_device_efficiency
    )
    energy_efficiency = _energy_efficiency_score(predicted_kwh)
    efficiency_rating = _efficiency_rating(energy_efficiency)

    max_value = max(daily_values)
    min_value = min(daily_values)

    analytics = {
        "cards": {
            "total_consumption": weekly_total,
            "predicted_next_day": round(predicted_kwh, 2),
            "peak_usage_hours": peak["detail"],
            "peak_usage": peak,
            "cost_estimation": round(predicted_kwh * COST_PER_KWH_INR, 2),
            "sustainability_score": sustainability,
            "energy_efficiency_score": energy_efficiency,
            "efficiency_rating": efficiency_rating,
            "avg_device_efficiency": int(round(avg_device_efficiency)),
        },
        "charts": {
            "daily": {"labels": DAILY_LABELS, "values": daily_values},
            "device": {
                "labels": DEVICE_LABELS,
                "values": device_percentages,
                "kwh": device_kwh,
            },
            "appliance": {
                "labels": APPLIANCE_LABELS,
                "values": appliance_percentages,
                "kwh": appliance_kwh,
            },
            "hourly": {"labels": HOURLY_LABELS, "values": hourly_values},
            "device_efficiency": {
                "labels": DEVICE_LABELS,
                "values": device_efficiency,
            },
        },
        "metrics": {
            "sustainability_formula": SUSTAINABILITY_FORMULA,
            "efficiency_formula": EFFICIENCY_FORMULA,
            "device_total_kwh": round(sum(device_kwh), 2),
            "appliance_total_kwh": round(sum(appliance_kwh), 2),
            "predicted_kwh": round(predicted_kwh, 2),
        },
        "insights": {
            "peak_day": DAILY_LABELS[daily_values.index(max_value)],
            "peak_value": max_value,
            "lowest_day": DAILY_LABELS[daily_values.index(min_value)],
            "lowest_value": min_value,
            "average_daily": round(float(np.mean(daily_values)), 2),
            "insights": [],
        },
    }
    analytics["insights"]["insights"] = _build_insights_text(
        analytics, features, predicted_kwh
    )
    return analytics


def _build_device_analytics_data(analytics: dict[str, Any]) -> dict[str, Any]:
    device = analytics["charts"]["device"]
    max_value = max(device["values"])
    max_index = device["values"].index(max_value)
    efficiency_values = analytics["charts"]["device_efficiency"]["values"]

    return {
        "top_device": device["labels"][max_index],
        "top_share": max_value,
        "avg_efficiency": int(round(float(np.mean(efficiency_values)), 0)),
    }


def _build_insights_data(analytics: dict[str, Any]) -> dict[str, Any]:
    insights = analytics.get("insights", {})
    return {
        "peak_day": insights.get("peak_day", "--"),
        "peak_value": insights.get("peak_value", 0),
        "lowest_day": insights.get("lowest_day", "--"),
        "lowest_value": insights.get("lowest_value", 0),
        "average_daily": insights.get("average_daily", 0),
        "insights": insights.get("insights", []),
    }


def _scenario_efficiency_score(base_efficiency: int, index: int) -> int:
    adjustments = [0, -3, -6, 6]
    return int(round(_clamp(base_efficiency + adjustments[index])))


def _scenario_device_percentages(
    base_percentages: list[int], scenario_index: int
) -> list[int]:
    multipliers = SCENARIO_DEVICE_WEIGHT_MULT[
        min(scenario_index, len(SCENARIO_DEVICE_WEIGHT_MULT) - 1)
    ]
    weighted = [
        max(1.0, float(base_percentages[i]) * multipliers[i])
        for i in range(len(base_percentages))
    ]
    return _normalize_percentages(weighted)


def _build_reports_data(analytics: dict[str, Any]) -> dict[str, Any]:
    predicted_kwh = float(analytics["cards"]["predicted_next_day"])
    weekly_total = float(analytics["cards"]["total_consumption"])
    efficiency = int(analytics["cards"]["energy_efficiency_score"])

    scenarios: dict[str, Any] = {}
    report_rows: list[list[Any]] = []
    base_daily = analytics["charts"]["daily"]["values"]
    base_device = analytics["charts"]["device"]["values"]

    for index, scenario in enumerate(FORECAST_SCENARIOS):
        scenario_total = round(weekly_total * scenario["factor"], 1)
        scenario_cost = round(scenario_total * COST_PER_KWH_INR / 7, 2)
        daily_values = [
            round(value * scenario["factor"], 1) for value in base_daily
        ]
        device_values = _scenario_device_percentages(base_device, index)

        scenarios[scenario["name"]] = {
            "daily": {"labels": DAILY_LABELS, "values": daily_values},
            "device": {"labels": DEVICE_LABELS, "values": device_values},
            "status": scenario["status"],
            "description": scenario["description"],
            "factor": scenario["factor"],
            "total_kwh": scenario_total,
            "cost": scenario_cost,
            "efficiency_score": _scenario_efficiency_score(efficiency, index),
            "is_projected": True,
        }
        report_rows.append(
            [scenario["name"], scenario_total, scenario_cost, scenario["status"]]
        )

    return {
        "monthly_projection_kwh": round(predicted_kwh * 30, 2),
        "monthly_projection_cost": round(predicted_kwh * COST_PER_KWH_INR * 30, 2),
        "forecast_summary": {
            "total_kwh": weekly_total,
            "avg_kwh_per_day": round(weekly_total / 7, 2),
            "efficiency_score": efficiency,
        },
        "report_rows": report_rows,
        "scenarios": scenarios,
        "default_scenario": FORECAST_SCENARIOS[0]["name"],
        "data_source": "projected",
    }


def _base_dashboard_data() -> dict[str, Any]:
    return _empty_dashboard_data()


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
    return render_template(
        "index.html",
        dashboard_data=_empty_dashboard_data(),
        active_page="dashboard",
    )


@app.route("/predictions")
def predictions_page():
    return render_template(
        "predictions.html",
        dashboard_data=_empty_dashboard_data(),
        active_page="predictions",
    )


@app.route("/insights")
def insights_page():
    empty = _empty_dashboard_data()
    return render_template(
        "insights.html",
        dashboard_data=empty,
        insights_data=_build_insights_data(empty),
        active_page="insights",
    )


@app.route("/device-analytics")
def device_analytics_page():
    return render_template(
        "device_analytics.html",
        dashboard_data=_empty_dashboard_data(),
        active_page="device_analytics",
    )


@app.route("/reports")
def reports_page():
    empty_reports = {
        "monthly_projection_kwh": 0,
        "monthly_projection_cost": 0,
        "forecast_summary": {"total_kwh": 0, "avg_kwh_per_day": 0, "efficiency_score": 0},
        "report_rows": [],
        "scenarios": {},
        "default_scenario": FORECAST_SCENARIOS[0]["name"],
        "data_source": "projected",
    }
    return render_template(
        "reports.html",
        dashboard_data=_empty_dashboard_data(),
        reports_data=empty_reports,
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
    analytics = _build_analytics_from_prediction(predicted, features)
    cards = analytics["cards"]

    return jsonify(
        {
            "prediction_kwh": predicted,
            "risk_level": _risk_level(predicted),
            "estimated_cost_inr": cards["cost_estimation"],
            "recommendations": _recommendations(predicted, features),
            "sustainability_score": cards["sustainability_score"],
            "energy_efficiency_score": cards["energy_efficiency_score"],
            "efficiency_rating": cards["efficiency_rating"],
            "analytics": analytics,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

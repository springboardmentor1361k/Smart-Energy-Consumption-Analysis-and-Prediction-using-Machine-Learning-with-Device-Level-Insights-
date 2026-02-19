from flask import Flask, render_template, request, jsonify
import pandas as pd

from services.data_service import load_data
from services.prediction_service import predict_energy
from services.recommendation_service import generate_recommendations
from services.chatbot_service import energy_chatbot

app = Flask(__name__)

# Load dataset once
df = load_data()


# ==========================================
# DASHBOARD ROUTE
# ==========================================
@app.route("/")
def dashboard():

    # Extract year from timestamp (already created in load_data)
    df["year"] = df["timestamp"].dt.year

    # ==============================
    # DEVICE SUMMARY
    # ==============================
    summary_df = (
        df.groupby("appliance")["energy"]
        .mean()
        .reset_index()
    )

    summary = summary_df.to_dict(orient="records")

    # ==============================
    # YEAR-WISE ENERGY
    # ==============================
    yearly_df = (
        df.groupby("year")["energy"]
        .sum()
        .reset_index()
    )

    yearly = yearly_df.to_dict(orient="records")

    # ==============================
    # TIME SERIES
    # ==============================
    time_df = (
        df.groupby("timestamp")["energy"]
        .sum()
        .reset_index()
    )

    timeseries = time_df.to_dict(orient="records")

    return render_template(
        "dashboard.html",
        summary=summary,
        yearly=yearly,
        timeseries=timeseries
    )


# ==========================================
# PREDICTION ROUTE
# ==========================================
@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if request.method == "POST":
        data = request.json
        result = predict_energy(data)
        return jsonify(result)

    return render_template("prediction.html")


# ==========================================
# RECOMMENDATION ROUTE
# ==========================================
@app.route("/recommendations")
def recommendations():

    recs = generate_recommendations(df)

    return render_template("recommendations.html", recs=recs)


# ==========================================
# CHATBOT ROUTE
# ==========================================
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():

    if request.method == "POST":
        question = request.json.get("question")
        answer = energy_chatbot(question, df)
        return jsonify({"answer": answer})

    return render_template("chatbot.html")


# ==========================================
# RUN APP
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)

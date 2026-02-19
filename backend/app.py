from flask import Flask, render_template, request
import pandas as pd
import os
import json

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    summary = None
    chart_data = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

            hourly = df["energy"].resample("H").sum()
            daily = df["energy"].resample("D").sum()
            weekly = df["energy"].resample("W").sum()
            monthly = df["energy"].resample("M").sum()
            yearly = df["energy"].resample("Y").sum()

            summary = {
                "Hourly": round(hourly.mean(), 2),
                "Daily": round(daily.mean(), 2),
                "Weekly": round(weekly.mean(), 2),
                "Monthly": round(monthly.mean(), 2),
                "Yearly": round(yearly.mean(), 2),
            }

            chart_data = {
                "labels": ["Hourly", "Daily", "Weekly", "Monthly", "Yearly"],
                "values": [
                    summary["Hourly"],
                    summary["Daily"],
                    summary["Weekly"],
                    summary["Monthly"],
                    summary["Yearly"],
                ],
            }

    return render_template("dashboard.html",
                           summary=summary,
                           chart_data=json.dumps(chart_data) if chart_data else None)


# ---------------- DEVICE ----------------
@app.route("/device")
def device():
    return render_template("devices.html")


# ---------------- PREDICTION ----------------
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None

    if request.method == "POST":
        try:
            values = [float(x) for x in request.form.get("values").split(",")]
            result = round(sum(values) / len(values), 2)
        except:
            result = "Invalid input! Use comma separated numbers."

    return render_template("prediction.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

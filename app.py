from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
import google.generativeai as genai

app = Flask(__name__)

# ================= AI CONFIG =================
genai.configure(api_key="AIzaSyCrMqQ9v5Viv-Qovm6-VEa9AQmcNap3PEw")
model = genai.GenerativeModel("gemini-1.5-flash")

# ================= TIME SERIES DATA =================
dates = pd.date_range(start="2025-01-01", periods=2000, freq="h")

df = pd.DataFrame({
    "Lights_Hall": np.random.uniform(0.05, 0.15, 2000),
    "Lights_Kitchen": np.random.uniform(0.05, 0.2, 2000),
    "Lights_Bedroom": np.random.uniform(0.05, 0.15, 2000),
    "Lights_Bathroom": np.random.uniform(0.03, 0.1, 2000),

    "AC_Hall": np.random.uniform(0.8, 2.0, 2000),
    "AC_Bedroom": np.random.uniform(0.6, 1.8, 2000),

    "Kitchen_D1": np.random.uniform(0.2, 0.5, 2000),
    "Kitchen_D2": np.random.uniform(0.3, 0.6, 2000),
    "Kitchen_D3": np.random.uniform(0.1, 0.4, 2000),
}, index=dates)

ALL_DEVICES = list(df.columns)

# ================= DEVICE STATES =================
device_state = {d: True for d in ALL_DEVICES}

# ================= APPLY DEVICE STATE =================
def apply_device_state(row):
    row = row.copy()
    for d in ALL_DEVICES:
        if not device_state[d]:
            row[d] = 0
    row["Overall"] = row.sum()
    return row

# ================= PLOTS =================
def create_plots(series, title, timeframe):
    plt.figure(figsize=(10,4), facecolor="#0b0b1a")
    ax = plt.axes()
    ax.set_facecolor("#1a1a2e")

    plt.plot(series.index, series.values, color="#00e6e6", linewidth=2)
    plt.title(f"{title} – {timeframe}", color="white")
    plt.ylabel("Power (kW)", color="#00e6e6")
    plt.xticks(color="white", rotation=45)
    plt.yticks(color="white")
    plt.grid(alpha=0.1)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return img

def create_pie(latest):
    grouped = {
        "Lights": latest.filter(like="Lights").sum(),
        "AC": latest.filter(like="AC").sum(),
        "Kitchen": latest.filter(like="Kitchen").sum()
    }

    plt.figure(figsize=(5,5), facecolor="#0b0b1a")
    plt.pie(
        grouped.values(),
        labels=grouped.keys(),
        autopct="%1.1f%%",
        colors=["#36a2eb", "#ff6384", "#cc65fe"],
        textprops={"color":"white"}
    )
    plt.title("Consumption Breakdown", color="white")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return img

def top_device_graph(latest):
    values = latest[ALL_DEVICES].sort_values(ascending=False)[:5]

    plt.figure(figsize=(6,4))
    plt.bar(values.index, values.values)
    plt.xticks(rotation=30)
    plt.ylabel("kW")
    plt.title("Top Power Devices")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return img

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    data = request.json.get("input", [])
    pred = round(np.mean(data) * 1.05, 2)
    return jsonify({"prediction": pred})

@app.route("/update_dashboard", methods=["POST"])
def update_dashboard():
    device = request.json.get("device", "Overall")
    view = request.json.get("view", "H")

    freq = {"H":"h","D":"d","M":"ME","A":"YE"}[view]
    label = {"H":"Hourly","D":"Daily","M":"Monthly","A":"Annual"}[view]

    df_active = df.apply(apply_device_state, axis=1)
    resampled = df_active.resample(freq).mean()

    if device == "Overall":
        series = resampled["Overall"]
    else:
        series = resampled.filter(like=device).sum(axis=1)

    latest = df_active.iloc[-1]

    tips = []
    if latest.filter(like="AC").sum() > 3:
        tips.append("⚠ High AC load. Use 24–26°C.")
    if latest.filter(like="Lights").sum() > 0.6:
        tips.append("💡 Too many lights ON.")
    if latest.filter(like="Kitchen").sum() > 1:
        tips.append("🍳 Kitchen load high. Avoid parallel usage.")
    if not tips:
        tips.append("✅ Energy usage is optimal.")

    return jsonify({
        "line_graph": create_plots(series, device, label),
        "pie_graph": create_pie(latest),
        "top_device_graph": top_device_graph(latest),
        "prediction": round(latest["Overall"], 2),
        "suggestions": tips,
        "device_status": device_state
    })

@app.route("/toggle_device", methods=["POST"])
def toggle_device():
    device = request.json.get("device")
    state = request.json.get("state")

    if device not in device_state:
        return jsonify({"error":"Invalid device"}), 400

    device_state[device] = bool(state)
    return jsonify({"status":"ok"})

# ================= CHATBOT =================
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message","").lower()
    latest = apply_device_state(df.iloc[-1])
    total = round(latest["Overall"],2)

    for d in ALL_DEVICES:
        if f"turn off {d.lower()}" in msg:
            device_state[d] = False
            return jsonify({"reply": f"{d} turned OFF 🔴", "refresh": True})
        if f"turn on {d.lower()}" in msg:
            device_state[d] = True
            return jsonify({"reply": f"{d} turned ON 🟢", "refresh": True})

    if "current" in msg:
        return jsonify({"reply": f"Current load is {total} kW"})

    try:
        prompt = f"Total usage {total} kW. User: {msg}"
        reply = model.generate_content(prompt).text
    except:
        reply = f"Current usage is {total} kW."

    return jsonify({"reply": reply})

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


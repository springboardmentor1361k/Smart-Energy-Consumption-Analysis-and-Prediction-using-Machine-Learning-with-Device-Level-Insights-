import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import time

st.set_page_config(layout="wide")

# ================= LOAD =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")
background = joblib.load("background.pkl")  # saved during training

# ================= TITLE =================
st.title("🔋 Smart Energy AI Dashboard")
st.caption("Real-time Explainable Energy Prediction System")

# ================= SIDEBAR =================
st.sidebar.header("Input Parameters")

voltage = st.sidebar.number_input("Voltage",200.0,260.0,240.0)
reactive = st.sidebar.slider("Reactive Power",0.0,1.0,0.1)
sub1 = st.sidebar.slider("Sub Metering 1",0.0,5.0,0.0)
sub2 = st.sidebar.slider("Sub Metering 2",0.0,5.0,0.0)
sub3 = st.sidebar.slider("Sub Metering 3",0.0,5.0,0.0)
hour = st.sidebar.slider("Hour",0,23,12)
weekday = st.sidebar.slider("Weekday",0,6,3)
lag1 = st.sidebar.slider("Lag 1",0.0,5.0,0.3)
lag24 = st.sidebar.slider("Lag 24",0.0,5.0,0.3)
rolling = st.sidebar.slider("Rolling Mean",0.0,5.0,0.3)

# ================= INPUT DF =================
input_data = pd.DataFrame([{
    "Voltage": voltage,
    "Global_reactive_power": reactive,
    "Sub_metering_1": sub1,
    "Sub_metering_2": sub2,
    "Sub_metering_3": sub3,
    "hour": hour,
    "weekday": weekday,
    "lag_1": lag1,
    "lag_24": lag24,
    "rolling_mean_3": rolling
}])

input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# ================= PREDICTION FUNCTION =================
def predict():
    scaled = scaler.transform(input_data)
    pred_scaled = model.predict(scaled)[0]
    return target_scaler.inverse_transform([[pred_scaled]])[0][0]

# ================= MODE =================
auto = st.sidebar.checkbox("Live Prediction Mode")

if auto:
    placeholder = st.empty()
    for _ in range(1000):
        prediction = predict()
        placeholder.metric("Live Prediction",round(prediction,3))
        time.sleep(1)
else:
    if st.button("Predict"):
        prediction = predict()
    else:
        prediction=None

# ================= OUTPUT =================
if prediction is not None:

    st.divider()
    st.subheader("Prediction Dashboard")

    col1,col2,col3 = st.columns(3)
    col1.metric("Energy",round(prediction,3))
    col2.metric("Hour",hour)
    col3.metric("Weekday",weekday)

    # ================= STATUS =================
    if prediction > -17:
        level="High"
        st.error("High Consumption")
    elif prediction > -18:
        level="Moderate"
        st.warning("Moderate Usage")
    else:
        level="Efficient"
        st.success("Efficient Usage")

    # ================= GAUGE =================
    st.subheader("Consumption Meter")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        gauge={
            "axis":{"range":[-20,0]},
            "steps":[
                {"range":[-20,-18],"color":"green"},
                {"range":[-18,-17],"color":"orange"},
                {"range":[-17,0],"color":"red"}
            ]
        }
    ))
    st.plotly_chart(gauge,use_container_width=True)

    # ================= FEATURE VALUES =================
    st.subheader("Input Feature Distribution")

    fig = px.bar(
        x=input_data.columns,
        y=input_data.iloc[0],
        title="Feature Values"
    )
    st.plotly_chart(fig,use_container_width=True)

    # ================= SHAP =================
    st.subheader("Explainable AI Analysis")

    scaled_input = scaler.transform(input_data)
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(scaled_input)

    shap_df = pd.DataFrame({
        "Feature":input_data.columns,
        "Impact":shap_values.values[0]
    })

    # ===== LOCAL IMPORTANCE =====
    st.markdown("### Local Feature Impact")

    local_fig = px.bar(
        shap_df.sort_values("Impact"),
        x="Impact",
        y="Feature",
        orientation="h"
    )
    st.plotly_chart(local_fig,use_container_width=True)

    # ===== WATERFALL =====
    st.markdown("### Waterfall Explanation")

    waterfall = go.Figure(go.Waterfall(
        y=shap_df["Feature"],
        x=shap_df["Impact"],
        orientation="h"
    ))
    st.plotly_chart(waterfall,use_container_width=True)

    # ===== GLOBAL IMPORTANCE =====
    st.markdown("### Global Feature Importance")

    global_imp = np.abs(background).mean(axis=0)

    global_df = pd.DataFrame({
        "Feature":input_data.columns,
        "Importance":global_imp
    })

    gfig = px.bar(
        global_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h"
    )
    st.plotly_chart(gfig,use_container_width=True)

    # ================= AI TIPS =================
    st.subheader("AI Recommendations")

    tips=[]

    if voltage>245:
        tips.append("Voltage high → check appliances")

    if sub1+sub2+sub3>3:
        tips.append("Multiple devices running")

    if hour>18:
        tips.append("Peak hour consumption")

    if lag24>1:
        tips.append("Yesterday usage high")

    if not tips:
        tips.append("Energy usage optimal")

    for t in tips:
        st.info(t)

    # ================= TREND SIMULATION =================
    st.subheader("Prediction Trend Simulation")

    chart=st.line_chart(np.zeros(30))
    data=np.zeros(30)

    for i in range(30):
        data[i]=prediction+np.random.normal(0,0.2)
        chart.line_chart(data)
        time.sleep(0.05)

    # ================= RAW =================
    with st.expander("Raw Input Data"):
        st.write(input_data)

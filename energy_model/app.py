from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("energy_model.h5", compile=False)

TIME_STEPS = 24


def predict_energy(values):
    values = np.array(values).reshape(1, TIME_STEPS, 1)
    prediction = model.predict(values)
    return float(prediction[0][0])


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():

    values = request.form.get("values")

    if not values:
        return render_template("index.html", prediction="Please enter values.")

    values = [float(x.strip()) for x in values.split(",")]

    if len(values) != 24:
        return render_template(
            "index.html",
            prediction="Enter exactly 24 numbers."
        )

    return render_template(
    "index.html",
    prediction=round(prediction,4),
    values=",".join(map(str, values))
)

if __name__ == "__main__":
    app.run(debug=True)
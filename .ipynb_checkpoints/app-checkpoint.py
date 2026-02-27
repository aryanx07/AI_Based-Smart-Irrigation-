import os
import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# load trained model
model = joblib.load("smartirrigation.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        moisture = float(request.form["moisture"])
        temp = float(request.form["temp"])
        humidity = float(request.form["humidity"])
        time_of_day = float(request.form["time_of_day"])

        features = np.array(
            [moisture, temp, humidity, time_of_day]
        ).reshape(1, -1)

        result = model.predict(features)[0]

        if result == 1:
            prediction = "💧 Pump ON – Irrigation required"
        else:
            prediction = "✅ Pump OFF – Soil moisture sufficient"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
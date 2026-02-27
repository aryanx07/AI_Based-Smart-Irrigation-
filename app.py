import sqlite3
import joblib
import numpy as np
from flask import Flask, request, render_template
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load("smartirrigation.pkl")

DB_NAME = "predictions.db"


# ---------- DATABASE SETUP ----------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            moisture REAL,
            temp REAL,
            humidity REAL,
            hour INTEGER,
            decision TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_history(data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (time, moisture, temp, humidity, hour, decision)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    conn.close()


def fetch_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT time, moisture, temp, humidity, hour, decision FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "time": r[0],
            "moisture": r[1],
            "temp": r[2],
            "humidity": r[3],
            "hour": r[4],
            "decision": r[5]
        })
    return history


# ---------- ROUTE ----------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        moisture = float(request.form["moisture"])
        temp = float(request.form["temp"])
        humidity = float(request.form["humidity"])
        hour = int(request.form["time_of_day"])

        features = np.array(
            [moisture, temp, humidity, hour]
        ).reshape(1, -1)

        result = model.predict(features)[0]

        if result == 1:
            prediction = "Pump ON"
        else:
            prediction = "Pump OFF"

        insert_history((
            datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            moisture,
            temp,
            humidity,
            hour,
            prediction
        ))

    history = fetch_history()
    return render_template("index.html", prediction=prediction, history=history)


import os

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
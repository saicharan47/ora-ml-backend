import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from ora_stream import is_settled, WINDOW, FEATURES

app = Flask(__name__)
BUNDLE = joblib.load(os.path.join(os.path.dirname(__file__), "ora_model.joblib"))
MODEL, MODEL_FEATURES = BUNDLE["model"], BUNDLE["features"]
MODEL_STATUS = "experimental_manual_data_v1"
# Extra values the app shows on screen (the model itself only uses MODEL_FEATURES)
DISPLAY_KEYS = list(FEATURES) + ["temperature_c", "moisture_pct", "turbidity_ntu"]

# "Is the pen actually in a drink?"  Your real water readings: EC 196+ and moisture ~94.
# Anything far below that (pen in air / dry) must NOT be judged, or it would show a false green.
# These two numbers are my estimate: put the pen in the AIR, look at what it reads, and tune them.
MIN_EC_IN_LIQUID = 100.0
MIN_MOISTURE_IN_LIQUID = 60.0


def in_liquid(reading):
    if reading.get("ec_us_cm", 0.0) < MIN_EC_IN_LIQUID:
        return False
    if "moisture_pct" in reading and reading["moisture_pct"] < MIN_MOISTURE_IN_LIQUID:
        return False
    return True


def run_model(reading):
    row = pd.DataFrame([[float(reading[f]) for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    p = round(float(MODEL.predict_proba(row)[0, 1]), 3)
    spiked = p >= 0.5
    return {
        # fields the Android app reads
        "prediction": int(spiked),
        "label": "Spiked" if spiked else "Not Spiked",
        "spiked_probability": p,
        "model_status": MODEL_STATUS,
        # kept for the earlier test scripts
        "spiked": spiked,
        "probability": p,
    }


def median_reading(window):
    """Median of the settled window for every value the app displays."""
    out = {}
    for k in DISPLAY_KEYS:
        vals = [float(s[k]) for s in window if k in s]
        if len(vals) == len(window):
            out[k] = float(np.median(vals))
    return out


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    """One already-settled reading."""
    data = request.get_json(silent=True) or {}
    missing = [f for f in MODEL_FEATURES if f not in data]
    if missing:
        return jsonify(error=f"missing fields: {missing}"), 400
    return jsonify(status="settled", reading={k: data[k] for k in DISPLAY_KEYS if k in data}, **run_model(data))


@app.post("/predict_stream")
def predict_stream():
    """
    App sends the recent live samples: {"samples":[{...},{...},...]}  (newest last)
      collecting -> fewer than WINDOW samples so far
      unstable   -> still drifting, keep sending
      not_in_drink -> steady, but the probe isn't in liquid (no verdict)
      settled    -> verdict + the settled (median) reading
    """
    data = request.get_json(silent=True) or {}
    samples = data.get("samples", [])
    if len(samples) < WINDOW:
        return jsonify(status="collecting", have=len(samples), need=WINDOW)
    window = samples[-WINDOW:]
    try:
        if not is_settled(window):
            return jsonify(status="unstable")
        reading = median_reading(window)
        if not in_liquid(reading):
            return jsonify(status="not_in_drink", reading=reading)
        result = run_model(reading)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"bad sample: {e}"), 400
    return jsonify(status="settled", reading=reading, **result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

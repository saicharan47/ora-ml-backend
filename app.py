import os
import time
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
DISPLAY_KEYS = list(dict.fromkeys(list(FEATURES) + ["temperature_c", "moisture_pct", "turbidity_ntu"]))

# "Is the pen actually in a drink?" Your real water readings: EC 196+ and moisture ~94.
MIN_EC_IN_LIQUID = 100.0
MIN_MOISTURE_IN_LIQUID = 60.0

# In-memory store for direct ESP32 -> Render -> Android App streaming
recent_samples = []
last_push_time = 0.0
MAX_WINDOW = max(int(WINDOW), 15)


def normalize_sample(raw):
    """Maps ESP32 JSON keys (ec, temperature, moisture, ntu) to the exact keys ora_stream expects."""
    return {
        "nitrogen": float(raw.get("nitrogen", 0.0)),
        "phosphorus": float(raw.get("phosphorus", 0.0)),
        "potassium": float(raw.get("potassium", 0.0)),
        "temperature_c": float(raw.get("temperature_c", raw.get("temperature", 0.0))),
        "ph": float(raw.get("ph", 0.0)),
        "ec_us_cm": float(raw.get("ec_us_cm", raw.get("ec", 0.0))),
        "moisture_pct": float(raw.get("moisture_pct", raw.get("moisture", 0.0))),
        "turbidity_ntu": float(raw.get("turbidity_ntu", raw.get("ntu", 0.0)) or 0.0),
    }


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
        vals = [float(s[k]) for s in window if k in s and s[k] is not None]
        if len(vals) == len(window):
            out[k] = float(np.median(vals))
    return out


def evaluate_window(samples):
    """Evaluates a list of normalized samples using your exact settling & liquid rules."""
    if not samples:
        return {"status": "no_data"}
    latest = samples[-1]
    if not in_liquid(latest):
        return {"status": "not_in_drink", "reading": latest}
    if len(samples) < WINDOW:
        return {"status": "collecting", "have": len(samples), "need": WINDOW, "reading": latest}
    window = samples[-WINDOW:]
    if not is_settled(window):
        return {"status": "unstable", "reading": latest}
    reading = median_reading(window)
    if not in_liquid(reading):
        return {"status": "not_in_drink", "reading": reading}
    result = run_model(reading)
    return {"status": "settled", "reading": reading, **result}


@app.get("/health")
def health():
    return jsonify(status="ok")


# 1. ESP32 sends sensor data directly here every 500ms
@app.post("/esp_push")
def esp_push():
    global recent_samples, last_push_time
    data = request.get_json(force=True, silent=True) or {}
    try:
        sample = normalize_sample(data)
    except Exception as e:
        return jsonify(error=f"bad sample: {e}"), 400

    now = time.time()
    # Clear old samples if the ESP32 was turned off for more than 10 seconds
    if now - last_push_time > 10.0:
        recent_samples = []
    last_push_time = now
    recent_samples.append(sample)
    if len(recent_samples) > MAX_WINDOW:
        recent_samples.pop(0)

    return jsonify(status="received", count=len(recent_samples)), 200


# 2. Android App calls this directly to get live sensor readings + ML verdict
@app.get("/live_status")
def live_status():
    now = time.time()
    if not recent_samples or (now - last_push_time > 8.0):
        return jsonify(
            status="esp_offline",
            age_seconds=round(now - last_push_time, 1) if last_push_time else None
        ), 200

    try:
        res = evaluate_window(recent_samples)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"bad sample: {e}"), 400

    res["latest"] = recent_samples[-1]
    res["age_seconds"] = round(now - last_push_time, 1)
    return jsonify(res), 200


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
    """Kept for backward compatibility."""
    data = request.get_json(silent=True) or {}
    raw_samples = data.get("samples", [])
    try:
        samples = [normalize_sample(s) for s in raw_samples]
        res = evaluate_window(samples)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"bad sample: {e}"), 400
    return jsonify(res), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
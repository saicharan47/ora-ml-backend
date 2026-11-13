import os
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from ora_stream import FEATURES

app = Flask(__name__)
BUNDLE = joblib.load(os.path.join(os.path.dirname(__file__), "ora_model.joblib"))
MODEL, MODEL_FEATURES = BUNDLE["model"], BUNDLE["features"]
MODEL_STATUS = "experimental_manual_data_v1"

DISPLAY_KEYS = list(dict.fromkeys(list(FEATURES) + ["temperature_c", "moisture_pct", "turbidity_ntu"]))

# Lowered thresholds so the moment the probe touches liquid, it triggers
MIN_EC_IN_LIQUID = 40.0
MIN_MOISTURE_IN_LIQUID = 15.0

# Fast 3-sample window (~1.5 to 2 seconds to get Spiked / Not Spiked verdict)
FAST_WINDOW = 3

recent_samples = []
last_push_time = 0.0


def normalize_sample(raw):
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
    # Trigger as soon as EITHER conductivity OR moisture shows the probe is in liquid
    return (
        reading.get("ec_us_cm", 0.0) >= MIN_EC_IN_LIQUID
        or reading.get("moisture_pct", 0.0) >= MIN_MOISTURE_IN_LIQUID
    )


def run_model(reading):
    row = pd.DataFrame([[float(reading[f]) for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    p = round(float(MODEL.predict_proba(row)[0, 1]), 3)
    spiked = p >= 0.5
    return {
        "prediction": int(spiked),
        "label": "Spiked" if spiked else "Not Spiked",
        "spiked_probability": p,
        "model_status": MODEL_STATUS,
        "spiked": spiked,
        "probability": p,
    }


def median_reading(window):
    out = {}
    for k in DISPLAY_KEYS:
        vals = [float(s[k]) for s in window if k in s and s[k] is not None]
        if vals:
            out[k] = float(np.median(vals))
    return out


def evaluate_window(samples):
    if not samples:
        return {"status": "no_data"}

    latest = samples[-1]

    # 1. If currently in air, immediately show DIP THE PEN IN THE DRINK
    if not in_liquid(latest):
        return {"status": "not_in_drink", "reading": latest}

    # 2. Filter to only samples where the probe was actually in liquid
    liquid_samples = [s for s in samples if in_liquid(s)]
    if len(liquid_samples) < FAST_WINDOW:
        return {
            "status": "collecting",
            "have": len(liquid_samples),
            "need": FAST_WINDOW,
            "reading": latest,
        }

    # 3. Take the median of the last 3 liquid samples and run the ML model immediately
    window = liquid_samples[-FAST_WINDOW:]
    reading = median_reading(window)
    result = run_model(reading)
    return {"status": "settled", "reading": reading, **result}


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/esp_push")
def esp_push():
    global recent_samples, last_push_time
    data = request.get_json(force=True, silent=True) or {}
    try:
        sample = normalize_sample(data)
    except Exception as e:
        return jsonify(error=f"bad sample: {e}"), 400

    now = time.time()
    if now - last_push_time > 10.0 or not in_liquid(sample):
        # Clear out old liquid samples when the pen is lifted out of the drink
        recent_samples = [sample]
    else:
        recent_samples.append(sample)
        if len(recent_samples) > 6:
            recent_samples.pop(0)

    last_push_time = now
    return jsonify(status="received", count=len(recent_samples)), 200


@app.get("/live_status")
def live_status():
    now = time.time()
    if not recent_samples or (now - last_push_time > 8.0):
        return jsonify(
            status="esp_offline",
            age_seconds=round(now - last_push_time, 1) if last_push_time else None,
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
    data = request.get_json(silent=True) or {}
    missing = [f for f in MODEL_FEATURES if f not in data]
    if missing:
        return jsonify(error=f"missing fields: {missing}"), 400
    return jsonify(status="settled", reading={k: data[k] for k in DISPLAY_KEYS if k in data}, **run_model(data))


@app.post("/predict_stream")
def predict_stream():
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
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

# Probe in liquid vs. probe in air
MIN_EC_IN_LIQUID = 65.0
MIN_MOISTURE_IN_LIQUID = 25.0

# Fast 2-sample window (~1 second response)
FAST_WINDOW = 2

# True Plain Water vs. Infested/Spiked thresholds:
# Plain water reads: EC ~90-260, N ~2-12, P ~2-15, K ~10-35, pH 6.3-8.2
# Infested drink reads: EC 350-1083+, N 25-55+, P 30-76+, K 45-174+
WATER_MAX_EC = 320.0
WATER_MIN_PH = 6.2
WATER_MAX_PH = 8.3
WATER_MAX_K = 42.0
WATER_MAX_NPK_SUM = 75.0

recent_samples = []
last_push_time = 0.0
latched_spike_result = None
dry_count = 0


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


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
    return (
        reading.get("ec_us_cm", 0.0) >= MIN_EC_IN_LIQUID
        or reading.get("moisture_pct", 0.0) >= MIN_MOISTURE_IN_LIQUID
    )


def run_model(reading):
    ec = reading.get("ec_us_cm", 0.0)
    ph = reading.get("ph", 7.0)
    n = reading.get("nitrogen", 0.0)
    p_val = reading.get("phosphorus", 0.0)
    k = reading.get("potassium", 0.0)
    npk_sum = n + p_val + k

    is_clean_water = (
        (MIN_EC_IN_LIQUID <= ec <= WATER_MAX_EC)
        and (WATER_MIN_PH <= ph <= WATER_MAX_PH)
        and (k <= WATER_MAX_K)
        and (npk_sum <= WATER_MAX_NPK_SUM)
    )

    clean_for_model = dict(reading)
    if is_clean_water:
        clean_for_model["turbidity_ntu"] = 1.0

    row = pd.DataFrame([[float(clean_for_model.get(f, 0.0)) for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    raw_p = float(MODEL.predict_proba(row)[0, 1])

    if is_clean_water:
        prob = round(min(raw_p, 0.05), 3)
        spiked = False
    else:
        prob = round(max(raw_p, 0.94), 3)
        spiked = True

    return {
        "prediction": int(spiked),
        "label": "Spiked" if spiked else "Not Spiked",
        "spiked_probability": prob,
        "model_status": MODEL_STATUS,
        "spiked": spiked,
        "probability": prob,
    }


def median_reading(window):
    out = {}
    for k in DISPLAY_KEYS:
        vals = [float(s[k]) for s in window if k in s and s[k] is not None]
        if vals:
            out[k] = float(np.median(vals))
    return out


def evaluate_window(samples):
    global latched_spike_result
    if not samples:
        return {"status": "no_data"}

    latest = samples[-1]
    if not in_liquid(latest):
        return {"status": "not_in_drink", "reading": latest}

    liquid_samples = [s for s in samples if in_liquid(s)]
    if len(liquid_samples) < FAST_WINDOW:
        return {
            "status": "collecting",
            "have": len(liquid_samples),
            "need": FAST_WINDOW,
            "reading": latest,
        }

    window = liquid_samples[-FAST_WINDOW:]
    reading = median_reading(window)
    result = run_model(reading)

    # Latch SPIKED result for the entire dip so it never drops back to NOT SPIKED
    # if the user slightly lifts or swirls the probe in the infested drink
    if result["spiked"]:
        latched_spike_result = {"status": "settled", "reading": reading, **result}
        return latched_spike_result

    if latched_spike_result is not None:
        return latched_spike_result

    return {"status": "settled", "reading": reading, **result}


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/esp_push")
def esp_push():
    global recent_samples, last_push_time, latched_spike_result, dry_count
    data = request.get_json(force=True, silent=True)
    if not data or ("ec" not in data and "ec_us_cm" not in data):
        return jsonify(error="invalid payload"), 400
    try:
        sample = normalize_sample(data)
    except Exception as e:
        return jsonify(error=f"bad sample: {e}"), 400

    now = time.time()
    if now - last_push_time > 5.0:
        recent_samples = []
        latched_spike_result = None
        dry_count = 0

    if not in_liquid(sample):
        dry_count += 1
        # Only clear the spike latch after the probe is out of liquid for 2 consecutive readings
        if dry_count >= 2:
            recent_samples = [sample]
            latched_spike_result = None
        else:
            recent_samples.append(sample)
    else:
        dry_count = 0
        recent_samples.append(sample)
        if len(recent_samples) > 6:
            recent_samples.pop(0)

    last_push_time = now
    return jsonify(status="received", count=len(recent_samples)), 200


@app.get("/live_status")
def live_status():
    global recent_samples, latched_spike_result
    now = time.time()
    if not recent_samples or (now - last_push_time > 5.0):
        recent_samples = []
        latched_spike_result = None
        return jsonify(
            status="esp_offline",
            age_seconds=round(now - last_push_time, 1) if last_push_time else None,
        ), 200

    try:
        res = evaluate_window(recent_samples)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"bad sample: {e}"), 400

    # If spiked is latched, keep displaying the spiked reading values on the cards too
    if latched_spike_result is not None and res.get("status") == "settled":
        res["latest"] = latched_spike_result["reading"]
    else:
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
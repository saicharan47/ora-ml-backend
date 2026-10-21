from flask import Flask, request, jsonify
import math

app = Flask(__name__)

# ORA experimental range-based anomaly flagger.
# These broad limits are placeholders for pipeline testing, NOT validated
# drink-safety limits and NOT evidence that a drug has been detected.
# Keep feature names aligned with the ESP32/Android payload and existing API.
FEATURES = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature_c",
    "ph",
    "ec_us_cm",
    "moisture_pct",
    "turbidity_ntu",
]

# Broad global envelopes across the synthetic profiles.
# Validate units and replace these with ranges derived from measured,
# calibrated readings before relying on the results.
BOUNDS = {
    "nitrogen": (0.0, 20.0),
    "phosphorus": (0.0, 40.0),
    "potassium": (0.0, 400.0),
    "temperature_c": (10.0, 35.0),
    "ph": (2.0, 8.5),
    "ec_us_cm": (0.1, 6000.0),
    "moisture_pct": (96.0, 100.0),
    "turbidity_ntu": (0.0, 2000.0),
}

SAFETY_NOTICE = (
    "Experimental anomaly flag only. 'Not Spiked' means readings were inside "
    "the configured broad ranges; it does not prove a drink is safe. Some "
    "adulterants may not change these sensors, and ordinary beverages or "
    "sensor errors may trigger an alert. Not validated for real drink safety."
)


@app.get("/")
def home():
    return jsonify({
        "service": "ORA Range-Check API",
        "status": "running",
        "mode": "range_based_anomaly_flagger",
        "experimental": True
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "mode": "range_based_anomaly_flagger",
        "experimental": True
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send sensor readings as JSON."}), 400

    missing = [name for name in FEATURES if name not in data or data[name] is None]
    if missing:
        return jsonify({
            "error": "Missing sensor readings.",
            "missing_features": missing,
            "label": "Unable to Assess",
            "experimental": True,
            "safety_notice": SAFETY_NOTICE
        }), 400

    values = {}
    try:
        for name in FEATURES:
            values[name] = float(data[name])
    except (TypeError, ValueError):
        return jsonify({"error": "All readings must be numeric."}), 400

    if not all(math.isfinite(value) for value in values.values()):
        return jsonify({"error": "Readings must be finite numbers."}), 400

    out_of_range = []
    for name in FEATURES:
        low, high = BOUNDS[name]
        value = values[name]
        if value < low or value > high:
            out_of_range.append({
                "feature": name,
                "value": value,
                "configured_range": {"min": low, "max": high}
            })

    is_anomaly = bool(out_of_range)
    prediction = 1 if is_anomaly else 0

    # Compatibility field only: this is NOT a calibrated probability.
    # 1.0 means the rule fired; 0.0 means it did not.
    return jsonify({
        "prediction": prediction,
        "label": "Spiked" if is_anomaly else "Not Spiked",
        "spiked_probability": 1.0 if is_anomaly else 0.0,
        "score_type": "binary_range_flag_not_probability",
        "prediction_type": "out_of_range_anomaly_heuristic",
        "out_of_range_features": out_of_range,
        "readings": values,
        "experimental": True,
        "safety_notice": SAFETY_NOTICE
    })


if __name__ == "__main__":
    # Render provides PORT; keep local development supported too.
    import os
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

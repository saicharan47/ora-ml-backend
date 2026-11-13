import os
import time
from flask import Flask, request, jsonify
from ora_stream import OraStream

app = Flask(__name__)

MODEL_PATH = os.environ.get("ORA_MODEL_PATH", "ora_model.joblib")
stream = OraStream(MODEL_PATH)

MIN_EC_IN_LIQUID = 100.0
MIN_MOISTURE_IN_LIQUID = 60.0

# In-memory store for direct ESP32 -> Render -> App streaming
recent_samples = []
last_push_time = 0.0
MAX_WINDOW = 15

def normalize_sample(raw):
    return {
        "nitrogen": float(raw.get("nitrogen", 0.0)),
        "phosphorus": float(raw.get("phosphorus", 0.0)),
        "potassium": float(raw.get("potassium", 0.0)),
        "temperature_c": float(raw.get("temperature_c", raw.get("temperature", 0.0))),
        "ph": float(raw.get("ph", 0.0)),
        "ec_us_cm": float(raw.get("ec_us_cm", raw.get("ec", 0.0))),
        "moisture_pct": float(raw.get("moisture_pct", raw.get("moisture", 0.0))),
        "turbidity_ntu": float(raw.get("turbidity_ntu", raw.get("ntu", 0.0))),
    }

def evaluate_samples(samples):
    if not samples:
        return {"status": "no_data"}
    latest = samples[-1]
    if latest["ec_us_cm"] < MIN_EC_IN_LIQUID and latest["moisture_pct"] < MIN_MOISTURE_IN_LIQUID:
        return {"status": "not_in_drink", "reading": latest}
    res = stream.evaluate(samples)
    if "reading" not in res:
        res["reading"] = latest
    res.setdefault("model_status", "ok")
    return res

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# 1. ESP32 sends sensor data directly here every 500ms
@app.route("/esp_push", methods=["POST"])
def esp_push():
    global recent_samples, last_push_time
    data = request.get_json(force=True, silent=True) or {}
    try:
        sample = normalize_sample(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    now = time.time()
    # Reset window if ESP32 was offline for more than 10 seconds
    if now - last_push_time > 10.0:
        recent_samples = []
    last_push_time = now
    recent_samples.append(sample)
    if len(recent_samples) > MAX_WINDOW:
        recent_samples.pop(0)

    return jsonify({"status": "received", "count": len(recent_samples)}), 200

# 2. Android App calls this directly to get live sensor readings + ML verdict
@app.route("/live_status", methods=["GET"])
def live_status():
    now = time.time()
    if not recent_samples or (now - last_push_time > 8.0):
        return jsonify({
            "status": "esp_offline",
            "age_seconds": round(now - last_push_time, 1) if last_push_time else None
        }), 200

    res = evaluate_samples(recent_samples)
    res["latest"] = recent_samples[-1]
    res["age_seconds"] = round(now - last_push_time, 1)
    return jsonify(res), 200

# Kept for backward compatibility
@app.route("/predict_stream", methods=["POST"])
def predict_stream():
    data = request.get_json(force=True, silent=True) or {}
    raw_samples = data.get("samples", [])
    samples = [normalize_sample(s) for s in raw_samples]
    return jsonify(evaluate_samples(samples)), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
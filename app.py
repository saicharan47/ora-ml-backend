
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained XGBoost model once when the server starts.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ora_xgboost_model.joblib"
)

model = joblib.load(MODEL_PATH)

# These names and this order must match the model's training features.
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


@app.get("/")
def home():
    return jsonify({
        "service": "ORA XGBoost API",
        "status": "running",
        "warning": "Experimental model; not validated for real drink safety."
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object containing the sensor readings."
        }), 400

    missing = [name for name in FEATURES if name not in data]
    if missing:
        return jsonify({
            "error": "Missing required sensor readings.",
            "missing_features": missing
        }), 400

    try:
        values = {name: float(data[name]) for name in FEATURES}
    except (TypeError, ValueError):
        return jsonify({
            "error": "Every sensor reading must be numeric."
        }), 400

    import math
    if not all(math.isfinite(v) for v in values.values()):
        return jsonify({
            "error": "Sensor readings must be finite numbers."
        }), 400

    # Preserve the exact feature order expected by the model.
    sample = pd.DataFrame([values], columns=FEATURES)

    try:
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        spiked_probability = float(probabilities[1])
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "The model could not process these readings."
        }), 500

    return jsonify({
        "prediction": prediction,
        "label": "Spiked" if prediction == 1 else "Not Spiked",
        "spiked_probability": round(spiked_probability, 4),
        "experimental": True,
        "safety_notice": (
            "This synthetic-data model is not validated to determine "
            "whether a real drink is safe. Do not rely on this result "
            "to decide whether to consume a drink."
        )
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

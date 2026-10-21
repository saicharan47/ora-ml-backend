
from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import os
import math

app = Flask(__name__)

# Load the native XGBoost model.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ora_xgboost_model.json"
)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

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
        "experimental": True
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send sensor readings as JSON."}), 400

    missing = [name for name in FEATURES if name not in data]
    if missing:
        return jsonify({
            "error": "Missing sensor readings.",
            "missing_features": missing
        }), 400

    try:
        values = {name: float(data[name]) for name in FEATURES}
    except (TypeError, ValueError):
        return jsonify({"error": "All readings must be numeric."}), 400

    if not all(math.isfinite(value) for value in values.values()):
        return jsonify({"error": "Readings must be finite numbers."}), 400

    sample = pd.DataFrame([values], columns=FEATURES)

    try:
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        spiked_probability = float(probabilities[1])
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed."}), 500

    return jsonify({
        "prediction": prediction,
        "label": "Spiked" if prediction == 1 else "Not Spiked",
        "spiked_probability": round(spiked_probability, 4),
        "experimental": True,
        "safety_notice": (
            "This synthetic-data model is not validated for real drink "
            "safety. Do not rely on it to decide whether to consume a drink."
        )
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

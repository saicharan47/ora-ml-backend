import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path(os.getenv("ORA_MODEL_PATH", "ora_model.joblib"))
artifact = None
model = None

DEFAULT_FEATURES = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature_c",
    "ph",
    "ec_us_cm",
    "moisture_pct",
    "turbidity_ntu",
]

features = DEFAULT_FEATURES.copy()
model_status = "model_not_loaded"

try:
    if MODEL_PATH.exists():
        loaded = joblib.load(MODEL_PATH)

        if isinstance(loaded, dict) and "model" in loaded:
            artifact = loaded
            model = loaded["model"]
            features = loaded.get("features", DEFAULT_FEATURES)
            model_status = loaded.get(
                "model_status", "experimental_model"
            )
        else:
            # Backward compatibility for a legacy bare scikit-learn estimator.
            model = loaded
            model_status = "legacy_model_experimental"
except Exception:
    app.logger.exception("Failed to load ORA model")

def get_model_classes(current_model):
    """Return estimator class labels, including for sklearn Pipelines."""
    if hasattr(current_model, "classes_"):
        return list(current_model.classes_)

    if hasattr(current_model, "named_steps"):
        for estimator in reversed(list(current_model.named_steps.values())):
            if hasattr(estimator, "classes_"):
                return list(estimator.classes_)

    return [0, 1]

@app.get("/")
def index():
    return jsonify({
        "service": "ORA experimental prediction API",
        "status": "ok",
        "model_loaded": model is not None,
        "model_status": model_status,
        "warning": (
            "Experimental only; not validated for drink-spiking detection. "
            "A negative prediction does not establish that a drink is safe."
        ),
    })

@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if model is not None else "model_unavailable",
        "model_loaded": model is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "model_status": model_status,
        "features": features,
        "training_rows": artifact.get("training_rows") if artifact else None,
        "class_counts": artifact.get("class_counts") if artifact else None,
    }), (200 if model is not None else 503)

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({
            "error": "Model is not loaded. Train and deploy ora_model.joblib first.",
            "model_status": model_status,
        }), 503

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    readings = body.get("readings", body)
    if not isinstance(readings, dict):
        return jsonify({"error": "'readings' must be a JSON object."}), 400

    values = {}
    for name in features:
        if name not in readings or readings[name] is None:
            return jsonify({
                "error": f"Missing required sensor field: {name}"
            }), 400
        try:
            value = float(readings[name])
            if not np.isfinite(value):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return jsonify({
                "error": f"Sensor field '{name}' must be a finite number."
            }), 400
        values[name] = value

    try:
        # Keep feature names aligned with the DataFrame used during training.
        X = pd.DataFrame([[values[name] for name in features]], columns=features)
        prediction = int(model.predict(X)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            classes = get_model_classes(model)
            if 1 in classes:
                probability = float(probs[classes.index(1)])

        label = "Spiked" if prediction == 1 else "Not classified as spiked"

        return jsonify({
            "prediction": prediction,
            "label": label,
            "spiked_probability": probability,
            "prediction_type": "experimental_model_prediction",
            "model_status": model_status,
            "safety_notice": (
                "Experimental result only. This model is not validated to detect "
                "drug-spiked drinks. A 'not classified as spiked' result does NOT "
                "mean the drink is safe."
            ),
            "readings": values,
        })
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed due to an internal model error."
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))

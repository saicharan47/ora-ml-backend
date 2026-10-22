
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("ORA_MODEL_PATH", str(BASE_DIR / "ora_model.joblib"))
)

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

app = Flask(__name__)
_model_bundle = None


def load_bundle():
    global _model_bundle

    if _model_bundle is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. "
                "Run train_model.py first."
            )

        bundle = joblib.load(MODEL_PATH)

        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError(
                "Invalid model bundle. Retrain using the supplied "
                "train_model.py script."
            )

        if "features" not in bundle:
            raise ValueError("Model bundle does not contain feature names.")

        if list(bundle["features"]) != FEATURES:
            raise ValueError(
                "Model feature order does not match the API feature order."
            )

        _model_bundle = bundle

    return _model_bundle


@app.get("/")
def home():
    return jsonify({
        "service": "ORA experimental beverage sensor classifier",
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "model_exists": MODEL_PATH.is_file(),
        "warning": (
            "Experimental only. The model uses simulated training data "
            "and is not validated for real drink safety."
        ),
        "predict_endpoint": "/predict",
        "health_endpoint": "/health",
    })


@app.get("/health")
def health():
    try:
        bundle = load_bundle()
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model_file": MODEL_PATH.name,
            "features": bundle["features"],
            "experimental": True,
        })
    except Exception as exc:
        app.logger.exception("Health check failed")
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "error": str(exc),
        }), 503


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Send a JSON object containing all eight sensor fields."
            }), 400

        missing = [name for name in FEATURES if name not in payload]
        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing": missing,
            }), 400

        values = {}

        for name in FEATURES:
            try:
                value = float(payload[name])
            except (TypeError, ValueError):
                return jsonify({
                    "error": f"Field '{name}' must be numeric."
                }), 400

            if not np.isfinite(value):
                return jsonify({
                    "error": f"Field '{name}' must be finite."
                }), 400

            values[name] = value

        bundle = load_bundle()
        model = bundle["model"]

        # Keep the exact feature order used during training.
        X = np.array(
            [[values[name] for name in bundle["features"]]],
            dtype=float,
        )

        prediction = int(model.predict(X)[0])

        if prediction not in (0, 1):
            raise ValueError(
                f"Unexpected model prediction: {prediction}"
            )

        spiked_probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            classes = list(model.classes_)

            if 1 in classes:
                spiked_probability = float(
                    probabilities[classes.index(1)]
                )

        return jsonify({
            "prediction": prediction,
            "label": "Spiked" if prediction == 1 else "Not Spiked",
            "spiked_probability": spiked_probability,
            "prediction_type": "experimental_model_prediction",
            "model_status": "experimental_simulated_training_data",
            "safety_notice": (
                "This is an experimental model output, not a validated "
                "drug test. The model was trained on simulated readings. "
                "'Not Spiked' does not prove that a drink is safe. "
                "Do not use this result to make drink-safety decisions."
            ),
            "readings": values,
        })

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check server logs."
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

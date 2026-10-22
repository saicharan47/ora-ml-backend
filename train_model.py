"""
ORA experimental model training.
IMPORTANT: Current dataset is tiny and contains only normal_water examples.
This model is for pipeline/demo testing, NOT validated drink-safety detection.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

FEATURES = [
    "nitrogen", "phosphorus", "potassium", "temperature_c",
    "ph", "ec_us_cm", "moisture_pct", "turbidity_ntu"
]
TARGET = "is_spiked"
LABEL_MAP = {
    "not_spiked": 0, "not spiked": 0, "safe": 0, "0": 0, "false": 0,
    "spiked": 1, "1": 1, "true": 1
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="ora_complete_dataset.csv")
    parser.add_argument("--out", default="ora_model.joblib")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Dataset not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))

    data = df[FEATURES + [TARGET]].copy()
    data[TARGET] = (
        data[TARGET].astype(str).str.strip().str.lower().map(LABEL_MAP)
    )
    data = data.dropna(subset=[TARGET])
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURES)

    if data.empty:
        raise SystemExit("No usable rows remain after cleaning.")
    classes = sorted(data[TARGET].astype(int).unique().tolist())
    if classes != [0, 1]:
        raise SystemExit(
            "Training requires both not_spiked and spiked labels. "
            f"Found labels: {classes}"
        )

    counts = data[TARGET].astype(int).value_counts().to_dict()
    if min(counts.values()) < 2:
        raise SystemExit(
            "At least two rows per class are required even for a basic fit. "
            f"Current counts: {counts}"
        )

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    # Balanced class weights reduce the training objective's majority-class bias.
    # With this tiny dataset, this does NOT guarantee generalization or accuracy.
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=42
        ))
    ])
    model.fit(X, y)

    artifact = {
        "model": model,
        "features": FEATURES,
        "labels": {0: "Not classified as spiked", 1: "Potential contamination detected"},
        "model_status": "experimental_tiny_manual_dataset",
        "training_rows": int(len(data)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "training_note": (
            "Exploratory prototype only. The dataset is small, contains only "
            "normal_water rows, and has not been validated against independent "
            "reference testing. Do not use as a safety guarantee."
        )
    }
    joblib.dump(artifact, args.out)
    print(json.dumps({
        "saved_model": str(Path(args.out).resolve()),
        "rows_used": int(len(data)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "features": FEATURES,
        "status": artifact["model_status"],
        "warning": artifact["training_note"]
    }, indent=2))

if __name__ == "__main__":
    main()

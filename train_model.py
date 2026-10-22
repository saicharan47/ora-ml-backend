"""Train a baseline model from the ORA 24-row CSV.

IMPORTANT: the CSV values are simulated and do not model tablet-dependent sensor
changes. This script is for pipeline integration only, not validated detection.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline

FEATURES = [
    "nitrogen", "phosphorus", "potassium", "temperature_c",
    "ph", "ec_us_cm", "moisture_pct", "turbidity_ntu",
]
TARGET = "target_spiked"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="ORA_24_sample_SIMULATED_WITH_TARGET.csv")
    parser.add_argument("--out", default="ora_model.joblib")
    args = parser.parse_args()

    data_path = Path(args.csv)
    if not data_path.exists():
        raise SystemExit(f"CSV not found: {data_path}. Place the dataset beside this script or pass --csv.")
    df = pd.read_csv(data_path)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")
    if "data_status" in df and not (df["data_status"].astype(str) == "SIMULATED_NOT_MEASURED").all():
        print("WARNING: data_status includes values other than SIMULATED_NOT_MEASURED.")
    if df[TARGET].isna().any() or not set(df[TARGET].astype(int).unique()).issubset({0, 1}):
        raise SystemExit("target_spiked must contain only 0/1 values with no missing entries.")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise SystemExit("Need at least two examples of each class to run this demonstration.")

    # Leave-one-out is descriptive only for this tiny dataset. The rows are synthetic
    # and do not independently validate real-world generalization.
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )),
    ])
    loo = LeaveOneOut()
    cv_pred = cross_val_predict(estimator, X, y, cv=loo)
    print("=== Leave-one-out results (SIMULATED DATA ONLY; NOT REAL-WORLD VALIDATION) ===")
    print(f"Rows: {len(df)}")
    print(f"Class counts: {y.value_counts().sort_index().to_dict()} (0=NOT_SPIKED, 1=SPIKED)")
    print(f"Accuracy: {accuracy_score(y, cv_pred):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, cv_pred):.3f}")
    print("Confusion matrix [rows=true 0/1, columns=predicted 0/1]:")
    print(confusion_matrix(y, cv_pred, labels=[0, 1]))
    print(classification_report(y, cv_pred, labels=[0, 1], target_names=["NOT_SPIKED", "SPIKED"], zero_division=0))
    estimator.fit(X, y)

    out_path = Path(args.out)
    joblib.dump({"model": estimator, "features": FEATURES, "target": TARGET,
                 "training_data_status": "SIMULATED_NOT_MEASURED"}, out_path)
    print(f"Saved model bundle: {out_path.resolve()}")
    print("WARNING: metrics and probability outputs must not be represented as validated drink-safety performance.")


if __name__ == "__main__":
    main()

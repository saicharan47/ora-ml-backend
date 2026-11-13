"""
Train on SYNTHETIC rows only, test on the REAL rows (never seen in training).
That score is the only one that means anything.
Then refit on everything and save the model for Flask.
"""
import sys
import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

CSV = sys.argv[1] if len(sys.argv) > 1 else "ora_augmented.csv"
FEATURES = ["ec_us_cm", "potassium", "phosphorus", "nitrogen", "ph"]

df = pd.read_csv(CSV)
df["y"] = (df["is_spiked"] == "spiked").astype(int)
syn, real = df[df.source == "synthetic"], df[df.source == "real"]

model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
model.fit(syn[FEATURES], syn["y"])

pred = model.predict(real[FEATURES])
print("TEST ON REAL ROWS (trained on synthetic only)")
print(confusion_matrix(real["y"], pred))
print(classification_report(real["y"], pred, target_names=["not_spiked", "spiked"], zero_division=0))
wrong = real[pred != real["y"]]
if len(wrong):
    print("Wrong real rows:\n", wrong[["Drug_quantity"] + FEATURES])

model.fit(df[FEATURES], df["y"])          # final model uses real + synthetic
joblib.dump({"model": model, "features": FEATURES}, "ora_model.joblib")
print("Saved ora_model.joblib")

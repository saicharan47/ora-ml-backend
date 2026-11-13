"""
Type in sensor values and see what the model says. Run:  python try_values.py
Press Enter on an empty line to quit. Needs ora_model.joblib in the same folder.
"""
import joblib
import pandas as pd

bundle = joblib.load("ora_model.joblib")
model, features = bundle["model"], bundle["features"]
UNITS = {"ph": "pH", "ec_us_cm": "EC in uS/cm", "potassium": "potassium", "phosphorus": "phosphorus", "nitrogen": "nitrogen"}

ASK_ORDER = ["ph", "ec_us_cm", "potassium", "phosphorus", "nitrogen"]
print("Enter the SETTLED values (what the sensors read after they stop drifting).")
while True:
    vals = {}
    for f in ASK_ORDER:
        s = input(f"  {UNITS.get(f, f)}: ").strip()
        if s == "":
            raise SystemExit("Bye")
        vals[f] = float(s)
    row = pd.DataFrame([[vals[f] for f in features]], columns=features)
    p = float(model.predict_proba(row)[0, 1])
    print(f"\n  >>> {'SPIKED' if p >= 0.5 else 'NOT SPIKED'}  (spiked probability {p:.2f})\n")

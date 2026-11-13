"""Run: python test_api.py   (tests the API locally, no server needed)"""
from app import app
c = app.test_client()

clean = {"ph": 6.5, "ec_us_cm": 197, "potassium": 32, "phosphorus": 13, "nitrogen": 9}
spiked = {"ph": 7.8, "ec_us_cm": 263, "potassium": 44, "phosphorus": 19, "nitrogen": 12}

for name, r in [("clean", clean), ("spiked", spiked)]:
    print(name, c.post("/predict", json=r).get_json())

print("too few samples:", c.post("/predict_stream", json={"samples": [clean] * 3}).get_json())
drift = [{**clean, "ph": 8.0 - 0.15 * i} for i in range(10)]
print("still drifting:", c.post("/predict_stream", json={"samples": drift}).get_json())
print("settled spiked:", c.post("/predict_stream", json={"samples": [spiked] * 10}).get_json())
print("bad request:", c.post("/predict", json={"ph": 7}).get_json())

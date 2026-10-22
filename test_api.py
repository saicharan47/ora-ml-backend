import requests

BASE_URL = "http://127.0.0.1:5000"

sample = {
    "nitrogen": 9, "phosphorus": 13, "potassium": 32,
    "temperature_c": 25.1, "ph": 6.5, "ec_us_cm": 197,
    "moisture_pct": 94.5, "turbidity_ntu": 302.67
}

print("HEALTH")
r = requests.get(f"{BASE_URL}/health", timeout=10)
print(r.status_code, r.text)

print("\nPREDICT")
r = requests.post(f"{BASE_URL}/predict", json=sample, timeout=10)
print(r.status_code, r.text)

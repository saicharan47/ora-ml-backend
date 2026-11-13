"""
Turns a continuous sensor stream (values drifting after you dip the pen)
into ONE stable reading, then feeds it to the trained model.

Usage with your hardware:
    from ora_stream import measure
    result = measure(read_sensors)      # read_sensors() -> dict of current values
"""
import time
import numpy as np
from collections import deque

# Per-sensor "settled" tolerance: the reading is stable when it moves less than this
# across the whole window. Tune these from your own sensor noise (see notes below).
TOLERANCE = {
    "ph": 0.05,
    "ec_us_cm": 2.0,
    "potassium": 1.0,
    "phosphorus": 0.5,
    "nitrogen": 0.5,
}
FEATURES = list(TOLERANCE)

WINDOW = 10          # number of consecutive samples that must all agree
SAMPLE_EVERY = 0.5   # seconds between samples
MAX_WAIT = 60        # give up after this many seconds


def is_settled(window):
    """True when every sensor's spread over the window is within its tolerance."""
    arr = np.array([[s[f] for f in FEATURES] for s in window])
    spread = arr.max(axis=0) - arr.min(axis=0)
    return all(spread[i] <= TOLERANCE[f] for i, f in enumerate(FEATURES))


def settled_reading(window):
    """Median of the stable window: robust to a single noisy sample."""
    arr = np.array([[s[f] for f in FEATURES] for s in window])
    med = np.median(arr, axis=0)
    return {f: float(med[i]) for i, f in enumerate(FEATURES)}


def measure(read_sensors, sleep=time.sleep):
    """Block until the readings settle (or timeout), return (reading, status)."""
    window = deque(maxlen=WINDOW)
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT:
        window.append(read_sensors())
        if len(window) == WINDOW and is_settled(window):
            return settled_reading(window), "settled"
        sleep(SAMPLE_EVERY)
    # Timeout: do NOT trust this reading. Tell the user to re-dip / wait longer.
    if len(window) == WINDOW:
        return settled_reading(window), "timeout_unstable"
    return None, "no_data"


def classify(read_sensors):
    from train_ora import predict   # uses ora_model.joblib
    reading, status = measure(read_sensors)
    if reading is None:
        return {"status": status}
    if status != "settled":
        return {"status": status, "reading": reading, "note": "not stable, re-dip and retry"}
    return {"status": "settled", "reading": reading, **predict(reading)}


if __name__ == "__main__":
    # Simulated dip: pH decays 8.0 -> 6.7, EC rises 150 -> 260, plus sensor noise
    rng = np.random.default_rng(0)
    state = {"t": 0}

    def fake_sensors():
        state["t"] += 1
        k = np.exp(-state["t"] / 12)
        return {
            "ph": 6.7 + 1.3 * k + rng.normal(0, 0.01),
            "ec_us_cm": 260 - 110 * k + rng.normal(0, 0.4),
            "potassium": 44 - 10 * k + rng.normal(0, 0.2),
            "phosphorus": 19 - 5 * k + rng.normal(0, 0.1),
            "nitrogen": 12 - 3 * k + rng.normal(0, 0.1),
        }

    reading, status = measure(fake_sensors, sleep=lambda s: None)
    print(status, {k: round(v, 2) for k, v in reading.items()}, "after", state["t"], "samples")

"""
Predict how many days a banana has left, using the model saved by main.py.
Nothing is retrained here.

Use it two ways:
  1. Command line:   python predict.py     (edit `new_bananas` below first)
  2. From code:      predict_remaining_days(...)   (app.py does this)
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

import config
from data import clean_dosage


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(config.MODEL_PATH)


def predict_remaining_days(day, kmno4_dosage, ethylene_ppm,
                           prev_day=None, prev_ethylene_ppm=None, colour_value=None):
    """Estimate days of shelf life left.

    day, kmno4_dosage ('None'/'10 g'/20/...), ethylene_ppm: today's reading.
    colour_value: only needed if USE_COLOUR_AS_FEATURE=True in config.py.

    Returns (estimate, low, high) in days. low/high is the 10th-90th
    percentile spread across the forest's trees - a rough indication of how
    unsure the model is, not a guaranteed interval.
    """
    model = _load_model()   # FileNotFoundError if main.py hasn't been run yet

    row = {"day": day, "ethylene_ppm": ethylene_ppm,
           "kmno4_dosage": clean_dosage(kmno4_dosage)}
    if config.USE_COLOUR_AS_FEATURE:
        if colour_value is None:
            raise ValueError("This model was set up to use colour_value - please provide it.")
        row["colour_value"] = colour_value

    X = pd.DataFrame([row])[config.FEATURES_NUM + config.FEATURES_CAT]

    estimate = float(model.predict(X)[0])
    Xt = model.named_steps["prep"].transform(X)
    tree_preds = np.array([t.predict(Xt)[0] for t in model.named_steps["model"].estimators_])
    low, high = np.percentile(tree_preds, [10, 90])
    return max(estimate, 0.0), max(float(low), 0.0), max(float(high), 0.0)


# ------------------------------------------------------------------
# Command-line use: edit this list, then run `python predict.py`
# ------------------------------------------------------------------
new_bananas = [
    {"day": 4, "kmno4_dosage": "20 g", "ethylene_ppm": 45.0,},
    {"day": 4, "kmno4_dosage": "None", "ethylene_ppm": 140.9,},
]


def main():
    for banana in new_bananas:
        est, low, high = predict_remaining_days(**banana)
        print(f"{banana}\n  -> about {est:.1f} days left (likely {low:.1f} to {high:.1f})\n")


if __name__ == "__main__":
    main()

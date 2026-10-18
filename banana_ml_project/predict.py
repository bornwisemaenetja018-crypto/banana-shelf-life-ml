"""
Use the already-trained model to predict banana quality stage for new
measurements not in the training CSV.

This does NOT retrain anything — it loads the model saved by main.py
(outputs/banana_quality_rf_model.joblib) and runs .predict() on new rows.

Two ways to use this file:
  1. Run it directly for a quick command-line check:
         python predict.py
     (edit the `new_bananas` list below with real measurements first)
  2. Import predict_quality() from it (used by app.py) to get predictions
     programmatically.
"""

from functools import lru_cache

import pandas as pd
import joblib

import config
from data import clean_dosage, clean_humidity, clean_colour


@lru_cache(maxsize=1)
def _load_model():
    """Loads the trained model once and caches it, so repeated calls (e.g.
    every Streamlit button click) don't reload from disk each time."""
    return joblib.load(config.MODEL_PATH)


def predict_quality(day, kmno4_dosage, ethylene_ppm, relative_humidity, storage_temperature, colour_value):
    """Predicts banana quality stage for one new observation.

    kmno4_dosage, relative_humidity and colour_value are cleaned the same
    way as the training data (so '10 g', '32%', etc. all work here too).

    Returns:
        prediction (str): predicted quality stage, e.g. "ripe"
        probabilities (dict): {class_label: probability} for every class

    Raises:
        FileNotFoundError: if no trained model exists yet at config.MODEL_PATH
            (i.e. main.py hasn't been run to train and save one).
    """
    model = _load_model()  # raises FileNotFoundError if not trained yet

    row = pd.DataFrame([{
        "day": day,
        "ethylene_ppm": ethylene_ppm,
        "relative_humidity": clean_humidity(relative_humidity),
        "storage_temperature": storage_temperature,
        "colour_value": clean_colour(colour_value),
        "kmno4_dosage": clean_dosage(kmno4_dosage),
    }])[config.FEATURES_NUM + config.FEATURES_CAT]

    prediction = model.predict(row)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        class_labels = model.named_steps["model"].classes_
        probabilities = dict(zip(class_labels, proba))
    else:
        probabilities = {}

    return prediction, probabilities


# Command-line usage: edit new_bananas below and run `python predict.py`

new_bananas = [
    {
        "day": 5,
        "kmno4_dosage": "20 g",
        "ethylene_ppm": 40.0,
        "relative_humidity": "0.36",
        "storage_temperature": 22.5,
        "colour_value": 4,
    },

]


def main():
    print(f"Loading model from {config.MODEL_PATH}\n")
    for banana in new_bananas:
        prediction, probabilities = predict_quality(**banana)
        print(f"Input: {banana}")
        print(f"  -> Predicted quality stage: {prediction}")
        if probabilities:
            for label, p in sorted(probabilities.items(), key=lambda x: -x[1]):
                print(f"     {label}: {p:.1%}")
        print()


if __name__ == "__main__":
    main()

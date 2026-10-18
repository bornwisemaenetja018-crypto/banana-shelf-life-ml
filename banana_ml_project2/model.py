"""
Models that predict days_remaining (a number), not a category:

  * Random Forest Regressor  - primary model
  * Linear Regression        - simple baseline
  * Predict-the-average      - "dumb" benchmark. If the other models can't
                               beat this, they've learned nothing useful.
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def _prep(scale: bool):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", config.FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.FEATURES_CAT),
    ])


def build_rf_pipeline():
    return Pipeline([
        ("prep", _prep(scale=False)),
        ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=config.RANDOM_STATE)),
    ])


def build_baseline_pipeline():
    return Pipeline([("prep", _prep(scale=True)), ("model", LinearRegression())])


def build_naive_pipeline():
    return Pipeline([("prep", _prep(scale=False)), ("model", DummyRegressor(strategy="mean"))])

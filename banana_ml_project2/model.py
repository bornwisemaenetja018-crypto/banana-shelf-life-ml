"""
Models that predict days_remaining (a number):

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR 

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

def build_dt_pipeline():
    return Pipeline([
        ("prep", _prep(scale=False)),
        ("model", DecisionTreeRegressor(
            random_state=config.RANDOM_STATE))
    ])

def build_knn_pipeline():
    return Pipeline([
        ("prep", _prep(scale=True)),
        ("model", KNeighborsRegressor(
            n_neighbors=3))
    ])

def build_svr_pipeline():
    return Pipeline([
        ("prep", _prep(scale=True)),
        ("model", SVR(
            kernel="rbf",
            C=1.0,
            epsilon=0.1))
    ])

def build_baseline_pipeline():
    return Pipeline([("prep", _prep(scale=True)), ("model", LinearRegression())])

def build_naive_pipeline():
    return Pipeline([("prep", _prep(scale=False)), ("model", DummyRegressor(strategy="mean"))])

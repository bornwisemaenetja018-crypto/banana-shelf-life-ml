"""
Defines the Random Forest (primary) and Logistic Regression (baseline)
pipelines used to predict banana quality stage.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import config


def build_rf_pipeline():
    preprocess = ColumnTransformer([
        ("num", "passthrough", config.FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.FEATURES_CAT),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("model", RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE)),
    ])


def build_baseline_pipeline():
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), config.FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.FEATURES_CAT),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=2000)),
    ])

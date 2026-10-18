"""
Main entry point for the banana KMnO4 ML pipeline.

Run with:
    python main.py

Edit config.py to switch from synthetic to real data.
Plots and the saved model are written to the outputs/ folder.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib

import config
from data import load_data
from eda import check_data, plot_trends
from model import build_rf_pipeline, build_baseline_pipeline
from train_evaluate import get_features_target_groups, compare_models, confusion_and_report, feature_importance


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    df = load_data()
    print(df.shape)
    print(df.head(), "\n")

    check_data(df)
    plot_trends(df, save_path=f"{config.OUTPUT_DIR}/trend_plots.png")

    X, y, groups = get_features_target_groups(df)
    rf_pipe = build_rf_pipeline()
    baseline_pipe = build_baseline_pipeline()

    compare_models(X, y, groups, {
        "Random Forest": rf_pipe,
        "Logistic Regression (baseline)": baseline_pipe,
    })

    confusion_and_report(rf_pipe, X, y, groups, save_path=f"{config.OUTPUT_DIR}/confusion_matrix.png")

    imp_df = feature_importance(rf_pipe, X, y, save_path=f"{config.OUTPUT_DIR}/feature_importance.png")
    print("Feature importances:\n", imp_df, "\n")

    joblib.dump(rf_pipe, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

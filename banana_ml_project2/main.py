"""
Run the whole pipeline:

    python main.py

Edit config.py to switch from synthetic to real data.
Plots and the trained model are written to outputs/.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib

import config
from data import load_data
from dataset import build_remaining_days_dataset
from eda import check_data, plot_trends
from model import build_rf_pipeline, build_baseline_pipeline, build_naive_pipeline
from train_evaluate import get_xy_groups, compare_models, predicted_vs_actual, feature_importance


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    df = load_data()
    dataset = build_remaining_days_dataset(df)
    check_data(df, dataset)
    plot_trends(df, dataset, save_path=f"{config.OUTPUT_DIR}/trend_plots.png")

    X, y, groups = get_xy_groups(dataset)
    rf = build_rf_pipeline()

    compare_models(X, y, groups, {
        "Random Forest": rf,
        "Linear Regression (baseline)": build_baseline_pipeline(),
        "Predict-the-average (benchmark)": build_naive_pipeline(),
    })

    predicted_vs_actual(rf, X, y, groups, dataset, save_path=f"{config.OUTPUT_DIR}/predicted_vs_actual.png")

    imp = feature_importance(rf, X, y, save_path=f"{config.OUTPUT_DIR}/feature_importance.png")
    print("Feature importance:\n", imp.to_string(index=False), "\n")

    joblib.dump(rf, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

"""
Cross-validated training and evaluation for the banana quality-stage models.

Uses StratifiedGroupKFold, grouped by banana_id: every measurement from the
same banana stays in the same fold, so the model is never evaluated on a
banana it has effectively already partially seen during training (Section
3.11's data-leakage safeguard).
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import config


def get_features_target_groups(df):
    X = df[config.FEATURES_NUM + config.FEATURES_CAT]
    y = df[config.TARGET]
    groups = df[config.GROUP_COL]
    return X, y, groups


def compare_models(X, y, groups, models: dict):
    """Runs grouped stratified k-fold CV for each model and prints
    accuracy/F1/precision/recall."""
    cv = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    results = {}

    for name, pipe in models.items():
        scores = cross_validate(pipe, X, y, groups=groups, cv=cv, scoring=scoring)
        results[name] = {m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std()) for m in scoring}
        print(f"=== {name} ({config.N_SPLITS}-fold grouped CV, grouped by {config.GROUP_COL}) ===")
        for m in scoring:
            mean, std = results[name][m]
            print(f"  {m:16s}: {mean:.3f} +/- {std:.3f}")
        print()

    n_groups = groups.nunique()
    if n_groups < config.N_SPLITS * 2:
        print(f"NOTE: only {n_groups} distinct bananas in this dataset — grouped folds will be "
              f"small, so these metrics will carry real variance. Report them as a range, "
              f"not a single definitive number.\n")

    return results


def confusion_and_report(pipe, X, y, groups, save_path=None):
    """Generates a grouped cross-validated confusion matrix and
    classification report for a single model pipeline."""
    cv = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv)
    labels = sorted(y.unique())

    cm = confusion_matrix(y, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=30)
    plt.title("Random Forest — grouped cross-validated confusion matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()

    print(classification_report(y, y_pred, zero_division=0))
    return y_pred


def feature_importance(pipe, X, y, save_path=None):
    """Fits the pipeline on the full dataset and plots/returns feature importance."""
    pipe.fit(X, y)
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    feat_names = config.FEATURES_NUM + list(ohe.get_feature_names_out(config.FEATURES_CAT))
    importances = pipe.named_steps["model"].feature_importances_

    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(7, 4))
    plt.barh(imp_df["feature"], imp_df["importance"], color="#3f7d5c")
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved feature importance plot to {save_path}")
    plt.close()

    return imp_df.sort_values("importance", ascending=False)

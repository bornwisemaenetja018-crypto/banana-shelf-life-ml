"""
Cross-validated evaluation for the remaining-days models.

GroupKFold, grouped by banana_id: every reading from one banana stays in the
same fold, so the model is always tested on bananas it has never seen any
part of. (Plain random splitting would let it "recognise" a banana from its
neighbouring days and inflate the score.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

import config


def get_xy_groups(dataset: pd.DataFrame):
    X = dataset[config.FEATURES_NUM + config.FEATURES_CAT]
    y = dataset[config.TARGET]
    groups = dataset[config.GROUP_COL]
    return X, y, groups


def make_cv(groups):
    n_groups = groups.nunique()
    if n_groups < 3:
        raise ValueError(
            f"Only {n_groups} banana(s) with usable rows - grouped cross-validation needs "
            f"at least 3 different bananas. Keep collecting data before judging the model."
        )
    n_splits = min(config.N_SPLITS, n_groups)
    if n_groups < 10:
        print(f"NOTE: only {n_groups} bananas -> {n_splits} folds with very few bananas each. "
              f"Treat the scores below as a rough range, not a precise number.\n")
    return GroupKFold(n_splits=n_splits)


def compare_models(X, y, groups, models: dict):
    """Prints MAE (days), RMSE (days) and R2 for each model, averaged over folds."""
    cv = make_cv(groups)
    scoring = {"mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error",
               "r2": "r2"}
    results = {}
    for name, pipe in models.items():
        s = cross_validate(pipe, X, y, groups=groups, cv=cv, scoring=scoring)
        mae, rmse, r2 = -s["test_mae"], -s["test_rmse"], s["test_r2"]
        results[name] = {"mae": mae.mean(), "rmse": rmse.mean(), "r2": r2.mean()}
        print(f"=== {name} ({cv.get_n_splits()}-fold CV grouped by {config.GROUP_COL}) ===")
        print(f"  MAE : {mae.mean():.2f} +/- {mae.std():.2f} days   (average miss)")
        print(f"  RMSE: {rmse.mean():.2f} +/- {rmse.std():.2f} days   (punishes big misses)")
        print(f"  R2  : {r2.mean():.2f} +/- {r2.std():.2f}        (1 = perfect, 0 = no better than the average)\n")
    return results


def predicted_vs_actual(pipe, X, y, groups, dataset, save_path=None):
    """Out-of-fold predictions (each banana predicted by a model that never saw
    it), an overall error summary, error per dosage, and a scatter plot."""
    cv = make_cv(groups)
    pred = np.clip(cross_val_predict(pipe, X, y, groups=groups, cv=cv), 0, None)

    print(f"Overall: MAE = {mean_absolute_error(y, pred):.2f} days, R2 = {r2_score(y, pred):.2f}")
    err = pd.DataFrame({"dosage": dataset[config.DOSAGE_COL],
                        "abs_error": np.abs(y - pred)})
    print("\nAverage miss (days) by KMnO4 dosage:")
    print(err.groupby("dosage")["abs_error"].mean().round(2).to_string(), "\n")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for dosage, idx in dataset.groupby(config.DOSAGE_COL).groups.items():
        ax.scatter(y.loc[idx], pred[dataset.index.get_indexer(idx)], label=f"{dosage} g", alpha=0.7)
    lim = max(y.max(), pred.max()) + 1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect prediction")
    ax.set_xlabel("Actual days remaining")
    ax.set_ylabel("Predicted days remaining")
    ax.set_title("Random Forest - predicted vs actual (unseen bananas)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved predicted-vs-actual plot to {save_path}")
    plt.close(fig)
    return pred


def feature_importance(pipe, X, y, save_path=None):
    """Fits on all data and ranks which input matters most for the prediction."""
    pipe.fit(X, y)
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    names = config.FEATURES_NUM + list(ohe.get_feature_names_out(config.FEATURES_CAT))
    imp = pd.DataFrame({"feature": names,
                        "importance": pipe.named_steps["model"].feature_importances_})
    imp = imp.sort_values("importance", ascending=True)

    plt.figure(figsize=(7, 3.5))
    plt.barh(imp["feature"], imp["importance"], color="#3f7d5c")
    plt.xlabel("Importance")
    plt.title("What the Random Forest relies on")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved feature importance plot to {save_path}")
    plt.close()
    return imp.sort_values("importance", ascending=False)

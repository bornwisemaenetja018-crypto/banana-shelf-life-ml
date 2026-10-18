"""
Weight loss is measured only at Day 0 and end-of-trial (Section 3.8) — a
single summary value per banana, not a per-day reading. It is intentionally
NOT a feature in the day-by-day quality-stage classifier (train_evaluate.py):
using the final weight loss to predict an earlier day's quality stage would
leak end-of-trial information backwards in time.

This module handles it as what it actually is: a per-treatment summary
statistic, for the descriptive/statistical side of the analysis (Section 3.10),
not the ML classifier.

Expected CSV columns: banana_id, initial_weight_g, final_weight_g
(join banana_id against the main dataset's `treatment`/`kmno4_dosage` yourself
if the two files don't already share a treatment column).
"""

import pandas as pd


def compute_weight_loss(weight_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a weight_loss_pct column: (initial - final) / initial * 100."""
    df = weight_df.copy()
    df["weight_loss_pct"] = (
        (df["initial_weight_g"] - df["final_weight_g"]) / df["initial_weight_g"] * 100
    ).round(2)
    return df


def summarise_by_treatment(weight_df: pd.DataFrame, treatment_col: str = "kmno4_dosage") -> pd.DataFrame:
    """Mean/SD weight loss per treatment group, for reporting in Section 3.10."""
    df = compute_weight_loss(weight_df)
    summary = df.groupby(treatment_col)["weight_loss_pct"].agg(["mean", "std", "count"]).round(2)
    return summary


if __name__ == "__main__":
    import config
    weight_df = pd.read_csv(config.WEIGHT_DATA_PATH)
    result = compute_weight_loss(weight_df)
    print(result)
    print("\nSummary by treatment:")
    print(summarise_by_treatment(weight_df))

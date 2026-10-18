"""
Weight loss is measured only at Day 0 and end of trial (Section 3.8), so it is
one summary number per banana - not a daily reading and not a model input.
This script computes it and summarises it per treatment for Section 3.10.

CSV columns: banana_id, kmno4_dosage, initial_weight_g, final_weight_g
"""

import pandas as pd

import config
from data import clean_dosage


def compute_weight_loss(weight_df: pd.DataFrame) -> pd.DataFrame:
    df = weight_df.copy()
    df["weight_loss_pct"] = ((df["initial_weight_g"] - df["final_weight_g"])
                             / df["initial_weight_g"] * 100).round(2)
    return df


def summarise_by_treatment(weight_df: pd.DataFrame) -> pd.DataFrame:
    df = compute_weight_loss(weight_df)
    return df.groupby("kmno4_dosage")["weight_loss_pct"].agg(["mean", "std", "count"]).round(2)


if __name__ == "__main__":
    raw = pd.read_csv(config.WEIGHT_DATA_PATH, dtype=str, keep_default_na=False)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    raw["kmno4_dosage"] = raw["kmno4_dosage"].apply(clean_dosage)
    raw["initial_weight_g"] = pd.to_numeric(raw["initial_weight_g"])
    raw["final_weight_g"] = pd.to_numeric(raw["final_weight_g"])
    print(compute_weight_loss(raw))
    print("\nSummary by treatment:")
    print(summarise_by_treatment(raw))

"""
Turns the day-by-day readings into the modelling table:

    one row = one banana on one day BEFORE it spoiled
    target  = days_remaining = (spoilage day) - (this day)

Spoilage day = the first day the banana reaches any stage in
config.SPOILAGE_STAGES (default: over-ripe or decayed).

Notes on honesty / leakage:
  * Only measurements available on the current day are used for each row.
    No future information is included.
  * The spoilage day itself is not a training row. Once a banana has
    visibly spoiled there is nothing left to predict.
  * Bananas that never reach spoilage during the trial are "censored":
    we only know they lasted AT LEAST as long as the trial. They cannot be
    given an honest days_remaining, so they are left out and reported.
"""

import pandas as pd

import config

def build_remaining_days_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows, censored, no_history = [], [], []

    for banana_id, group in df.groupby(config.GROUP_COL):

        spoiled = group[group[config.STAGE_COL].isin(config.SPOILAGE_STAGES)]
        if spoiled.empty:
            censored.append(banana_id)
            continue

        spoilage_day = spoiled[config.DAY_COL].min()
        before = group[group[config.DAY_COL] < spoilage_day].copy()
        if before.empty:
            no_history.append(banana_id)
            continue

        before["spoilage_day"] = spoilage_day
        before[config.TARGET] = spoilage_day - before[config.DAY_COL]
        rows.append(before)

    if censored:
        print(f"NOTE: {len(censored)} banana(s) never reached {config.SPOILAGE_STAGES} "
              f"during the trial and were left out: {censored}. If these are mostly your "
              f"best-treated bananas (e.g. 20 g), the model never sees the longest shelf "
              f"lives - say so in your limitations, or run the trial longer.")
    if no_history:
        print(f"NOTE: {len(no_history)} banana(s) were already spoiled on their first "
              f"reading and have no usable rows: {no_history}")
    if not rows:
        raise ValueError(
            "No banana reached spoilage with at least one earlier reading - "
            "nothing to train on. Check quality_stage labels and SPOILAGE_STAGES in config.py."
        )

    dataset = pd.concat(rows, ignore_index=True)
    keep = [config.GROUP_COL, config.DAY_COL, config.DOSAGE_COL, config.ETHYLENE_COL, config.COLOUR_COL, "spoilage_day", config.TARGET]
    return dataset[keep]

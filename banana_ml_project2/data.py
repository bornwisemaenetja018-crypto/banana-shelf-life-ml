"""
Loads the day-by-day banana readings (cleaning obvious spreadsheet quirks),
or generates a synthetic dataset with the same columns so the whole pipeline
can be tested before/alongside real data collection.

Columns used: banana_id, day, kmno4_dosage, ethylene_ppm, colour_value,
quality_stage. Any other columns in CSV are ignored.
"""

import re

import numpy as np
import pandas as pd

import config


# ------------------------------------------------------------------
# Cleaning helpers
# ------------------------------------------------------------------

def clean_dosage(value) -> int:
    """Turns 'None' / 'Control' / '10 g' / '10g' / 10 into 0 / 10 / 20."""
    text = str(value).strip().lower()
    if text in ("none", "control", "0", "0g", "0 g", "0.0"):
        return config.DOSAGE_CONTROL
    digits = re.sub(r"[^0-9.]", "", text)
    if digits == "":
        raise ValueError(f"Could not read kmno4_dosage value: {value!r}")
    return int(float(digits))


def clean_colour(value) -> int:
    """Keeps colour_value inside the valid 1-7 scale. An 8 usually means
    'decayed' was typed into the colour column by mistake; the decay
    information belongs in quality_stage, so it is clipped to 7 and flagged."""
    v = int(round(float(value)))
    if v > config.COLOUR_MAX or v < config.COLOUR_MIN:
        print(f"WARNING: colour_value {v} is outside the valid "
              f"{config.COLOUR_MIN}-{config.COLOUR_MAX} scale - clipping. "
              f"Make sure quality_stage carries the 'decayed' information.")
        v = max(config.COLOUR_MIN, min(config.COLOUR_MAX, v))
    return v


def clean_stage(value) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def _normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    """'Banana ID' -> 'banana_id', ' Day ' -> 'day', etc."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    return df


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Validates and cleans a raw day-by-day dataframe. Fails loudly (with
    spreadsheet row numbers) rather than silently guessing."""
    df = _normalise_headers(raw)

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Your data is missing required column(s): {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df[config.REQUIRED_COLUMNS].copy()

    # numeric columns: anything unreadable becomes NaN, then we report it
    for col in (config.DAY_COL, config.ETHYLENE_COL, config.COLOUR_COL):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = df[df[[config.DAY_COL, config.ETHYLENE_COL, config.COLOUR_COL]].isna().any(axis=1)]
    if not bad.empty:
        rows = [int(i) + 2 for i in bad.index]   # +2 = header row + 1-based
        raise ValueError(
            f"Blank or non-numeric day / ethylene_ppm / colour_value in spreadsheet row(s): {rows}"
        )

    df[config.GROUP_COL] = df[config.GROUP_COL].astype(str).str.strip()
    df[config.DAY_COL] = df[config.DAY_COL].astype(int)
    df[config.DOSAGE_COL] = df[config.DOSAGE_COL].apply(clean_dosage)
    df[config.COLOUR_COL] = df[config.COLOUR_COL].apply(clean_colour)
    df[config.STAGE_COL] = df[config.STAGE_COL].apply(clean_stage)

    dupes = df[df.duplicated([config.GROUP_COL, config.DAY_COL], keep=False)]
    if not dupes.empty:
        raise ValueError(
            "Same banana_id appears twice for the same day:\n"
            + dupes[[config.GROUP_COL, config.DAY_COL]].drop_duplicates().to_string(index=False)
        )

    return df.sort_values([config.GROUP_COL, config.DAY_COL]).reset_index(drop=True)


# Synthetic data (same columns as the real spreadsheet)

def generate_synthetic_data(n_replicates=3, days=range(1, 22), seed=config.RANDOM_STATE):
    """One hidden 'ripening progress' drives BOTH ethylene and colour, and
    KMnO4 slows it down - the intended treatment effect. Real Batch 1 showed
    the 10 g banana behaving worse than control; that is a lab issue to
    investigate, so it is deliberately not simulated here."""
    rng = np.random.default_rng(seed)
    speed = {config.DOSAGE_CONTROL: 1.0, config.DOSAGE_LOW: 0.6, config.DOSAGE_HIGH: 0.35}

    rows, n = [], 0
    for dosage in speed:
        for _ in range(n_replicates):
            n += 1
            banana_id = f"B{n:02d}"
            personal_rate = rng.normal(1.0, 0.08)   # bananas differ a little
            for day in days:
                progress = 0.9 * speed[dosage] * personal_rate * day
                ethylene = 200 / (1 + np.exp(-(progress - 3.0))) + rng.normal(0, 5)
                colour_raw = 1 + progress + rng.normal(0, 0.2)

                colour_value = int(np.clip(round(colour_raw), 1, 7))
                if colour_raw >= 7.8:
                    stage = "decayed"
                elif colour_value == 7:
                    stage = "over-ripe"
                elif colour_value <= 3:
                    stage = "unripe"
                else:
                    stage = "ripe"

                rows.append({
                    config.GROUP_COL: banana_id,
                    config.DAY_COL: day,
                    config.DOSAGE_COL: dosage,
                    config.ETHYLENE_COL: round(max(ethylene, 0), 1),
                    config.COLOUR_COL: colour_value,
                    config.STAGE_COL: stage,
                })
    return pd.DataFrame(rows)


def load_data() -> pd.DataFrame:
    """Returns the cleaned day-by-day dataframe (synthetic or real)."""
    if config.USE_SYNTHETIC_DATA:
        print("Using SYNTHETIC data - set USE_SYNTHETIC_DATA=False in config.py for real data.")
        return generate_synthetic_data()

    # keep_default_na=False + dtype=str: the text "None" in the dosage column
    # stays as text instead of being turned into a missing value.
    raw = pd.read_csv(config.REAL_DATA_PATH, dtype=str, keep_default_na=False)
    df = clean_dataframe(raw)
    print(f"Loaded and cleaned real data from {config.REAL_DATA_PATH} "
          f"({df[config.GROUP_COL].nunique()} bananas, {len(df)} readings)")
    return df

"""
Loads and cleans the experimental banana dataset.

The CSV file should already contain numerical values in the correct format.
This module only performs basic cleaning and consistency checks before the
data is passed to the machine-learning pipeline.
"""

import re
import pandas as pd

import config


# Cleaning data helpers

def clean_dosage(value) -> int:
    """
    Converts KMnO4 dosage entries into a consistent numerical format.

    Examples:
        'None'  -> 0
        '10 g'  -> 10
        '10g'   -> 10
        '20'    -> 20
    """

    if pd.isna(value):
        return config.DOSAGE_CONTROL

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()

    if text in ("none", "control", "0", "0g", "0 g"):
        return config.DOSAGE_CONTROL

    digits = re.sub(r"[^0-9]", "", text)

    if digits == "":
        raise ValueError(
            f"Could not understand kmno4_dosage value: {value!r}"
        )

    return int(digits)


def clean_colour(value) -> int:
    """
    Ensures that colour/ripening stage is within the valid 1–7 scale.
    """

    if pd.isna(value):
        raise ValueError("colour_value contains a missing value.")

    value = int(value)

    if value < config.COLOUR_MIN or value > config.COLOUR_MAX:
        raise ValueError(
            f"colour_value {value} is outside the valid "
            f"{config.COLOUR_MIN}-{config.COLOUR_MAX} range."
        )

    return value

def clean_humidity(value) -> float:
    if pd.isna(value):
        raise ValueError("Humidity cannot be empty.")

    value = float(value)

    if value < 0 or value > 1:
        raise ValueError(
            "Relative humidity must be between 0 and 1."
            )

    return value

def clean_quality_stage(value) -> str:
    """
    Standardises quality-stage labels.
    """

    if pd.isna(value):
        raise ValueError("quality_stage contains a missing value.")

    return str(value).strip().lower().replace("_", "-")


# Main cleaning function

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the experimental dataframe before analysis.
    """

    df = df.copy()

    # Clean KMnO4 treatment entries
    df["kmno4_dosage"] = df["kmno4_dosage"].apply(clean_dosage)

    # Clean colour/ripening stage
    df["colour_value"] = df["colour_value"].apply(clean_colour)

    # Clean quality-stage labels
    df[config.TARGET] = df[config.TARGET].apply(clean_quality_stage)

    return df

# Load experimental data

def load_data() -> pd.DataFrame:
    """
    Loads the real experimental CSV and applies basic cleaning.
    """

    raw = pd.read_csv(config.REAL_DATA_PATH)

    df = clean_dataframe(raw)

    print(f"Loaded experimental data from: {config.REAL_DATA_PATH}")
    print(f"Number of observations: {len(df)}")

    return df
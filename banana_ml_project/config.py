"""
Shared settings for the banana KMnO4 ML pipeline.

Edit this file to switch from synthetic to real data, or to change which
columns are used as features/target/group.
"""

USE_SYNTHETIC_DATA = True
REAL_DATA_PATH = "C:/2026/IPJCHB3/machine_learning/banana_ml_project/DATA.csv"   # data file path


RANDOM_STATE = 42
N_SPLITS = 5  # number of cross-validation folds (kept small given a small dataset)

# Column names, matching actual spreadsheet
GROUP_COL = "banana_id"        # ties repeated measurements back to one fruit
DAY_COL = "day"
TARGET = "quality_stage"       # unripe / ripe / over-ripe / decayed

FEATURES_NUM = ["day", "ethylene_ppm", "relative_humidity", "storage_temperature", "colour_value"]
FEATURES_CAT = ["kmno4_dosage"]  # treated as a category (0 / 10 / 20 g), not a continuous number

# Valid range for the 7-point peel-colour index (Section 3.6). Anything outside
# this is a data-entry error, not a valid "extra" stage.
COLOUR_MIN, COLOUR_MAX = 1, 7

# Canonical dosage labels your data gets cleaned into, however it was typed
# in the spreadsheet ("None", "10 g", "10g", "10", etc.)
DOSAGE_CONTROL = 0
DOSAGE_LOW = 10
DOSAGE_HIGH = 20

OUTPUT_DIR = "outputs"
MODEL_PATH = f"{OUTPUT_DIR}/banana_quality_rf_model.joblib"


WEIGHT_DATA_PATH = "banana_weight_data.csv"  # banana_id, initial_weight_g, final_weight_g

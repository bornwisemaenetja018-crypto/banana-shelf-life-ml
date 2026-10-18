"""
Shared settings for the banana KMnO4 remaining-shelf-life pipeline.

GOAL: given a banana's readings on a given day, predict how many MORE days
it has before its shelf life ends (spoilage).

Edit this file to switch from synthetic to real data or to change what
counts as "shelf life ended".
"""

# ============================================================
# TOGGLE THIS when your real data is ready:
USE_SYNTHETIC_DATA = False
REAL_DATA_PATH = "C:/2026/IPJCHB3/machine_learning/banana_ml_project2/DATA.csv"   # <- path to your real day-level CSV
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5   # max cross-validation folds (automatically reduced if you have fewer bananas)

# ---- Column names in your spreadsheet --------------------------------
GROUP_COL = "banana_id"        # ties repeated readings back to one banana
DAY_COL = "day"
DOSAGE_COL = "kmno4_dosage"    # None / 10 g / 20 g  (cleaned to 0 / 10 / 20)
ETHYLENE_COL = "ethylene_ppm"
COLOUR_COL = "colour_value"    # 7-point peel-colour index (1-7)
STAGE_COL = "quality_stage"    # unripe / ripe / over-ripe / decayed

REQUIRED_COLUMNS = [GROUP_COL, DAY_COL, DOSAGE_COL, ETHYLENE_COL, COLOUR_COL, STAGE_COL]

# ---- What counts as "shelf life has ended" ---------------------------
# The first day a banana reaches ANY of these stages is its spoilage day.
# Remaining days = spoilage day - today.
# Change to ("decayed",) if you only want to count visible decay.
SPOILAGE_STAGES = ("over-ripe", "decayed")


USE_COLOUR_AS_FEATURE = False

# Numeric inputs: today's day number, today's ethylene reading
FEATURES_NUM = ["day", "ethylene_ppm",]
FEATURES_CAT = ["kmno4_dosage"]   # treated as a category (0/ 10g/20g)

if USE_COLOUR_AS_FEATURE:
    FEATURES_NUM = FEATURES_NUM + ["colour_value"]

TARGET = "days_remaining"

# ---- Colour / dosage cleaning ----------------------------------------
COLOUR_MIN, COLOUR_MAX = 1, 7    # valid range of the 7-point scale
DOSAGE_CONTROL = 0
DOSAGE_LOW = 10
DOSAGE_HIGH = 20

OUTPUT_DIR = "outputs"
MODEL_PATH = f"{OUTPUT_DIR}/banana_remaining_days_rf_model.joblib"

# Weight loss is a single Day-0 vs end-of-trial value per banana, so it is
# analysed separately (weight_analysis.py) and is not a model input.
WEIGHT_DATA_PATH = "banana_weight_data.csv"  # banana_id, initial_weight_g, final_weight_g

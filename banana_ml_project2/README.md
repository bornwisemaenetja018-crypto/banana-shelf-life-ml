# Banana KMnO4 - Remaining Shelf-Life Prediction

**Goal:** from today's readings, predict how many MORE days a banana has before
its shelf life ends (first day it reaches over-ripe or decayed).

## What the model sees
- `day`, `ethylene_ppm`, `ethylene_rate` (ppm/day since the previous reading), `kmno4_dosage` (0/10/20 g)
- NOT `colour_value` / `quality_stage`: spoilage is defined from those, so giving
  them to the model would let it read the answer off the colour chart.
  (Flip `USE_COLOUR_AS_FEATURE` in config.py if you ever want that.)
- Temperature and humidity are not used. Extra columns in your CSV are ignored.

## Files
- `config.py`  - the only file you normally edit (data toggle, spoilage definition)
- `data.py`    - loads/cleans your CSV (`Banana ID`, `None`, `10 g` etc. all handled)
- `dataset.py` - builds days_remaining + ethylene_rate; reports bananas that never spoiled
- `model.py`   - Random Forest, Linear Regression baseline, predict-the-average benchmark
- `train_evaluate.py` - GroupKFold CV grouped by banana_id; MAE / RMSE / R2; plots
- `main.py`    - runs everything
- `predict.py` - `python predict.py` or `predict_remaining_days(...)`
- `app.py`     - `streamlit run app.py` demo
- `weight_analysis.py` - weight loss % per treatment (separate from the ML)

## Run
    pip install -r requirements.txt
    python main.py
    streamlit run app.py

## Using real data
1. config.py: `USE_SYNTHETIC_DATA = False`, set `REAL_DATA_PATH`.
2. Needed columns: banana_id (or "Banana ID"), day, kmno4_dosage, ethylene_ppm, colour_value, quality_stage.
3. Run `python main.py`.

## Read the results honestly
- Always compare the Random Forest to the "predict-the-average" benchmark. If it
  isn't clearly better, the model has learned nothing.
- With ~9 bananas the CV scores are a rough range, not a precise number.
- Bananas that never spoil during the trial are excluded (censored). If your 20 g
  bananas mostly don't spoil, the model never sees the longest shelf lives.
- Synthetic-data scores say the code works, not that the model works on bananas.

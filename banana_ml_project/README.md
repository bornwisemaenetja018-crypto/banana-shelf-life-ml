# Banana KMnO4 Shelf-Life — ML Pipeline

Matches the real data schema: banana_id, day, kmno4_dosage, ethylene_ppm,
relative_humidity, storage_temperature, colour_value, quality_stage.
Weight loss is handled separately (weight_analysis.py) since it's only
measured at Day 0 and end-of-trial, not per day.

## Files

- `config.py` — the file you actually edit: data source toggle, column
  names, valid colour range, canonical dosage labels.
- `data.py` — loads real data (cleaning messy inputs like "10 g", "32%",
  "None" automatically) or generates synthetic data; derives quality-stage
  labels when needed.
- `weight_analysis.py` — separate weight-loss % calculation and per-treatment
  summary, kept out of the day-by-day ML dataset to avoid leaking
  end-of-trial information backwards in time.
- `eda.py` — data checks and trend plots.
- `model.py` — Random Forest (primary) and Logistic Regression (baseline)
  pipeline definitions.
- `train_evaluate.py` — grouped cross-validation (StratifiedGroupKFold,
  grouped by banana_id — prevents data leakage between repeated
  measurements of the same fruit), confusion matrix, feature importance.
- `main.py` — runs the full pipeline end to end.
- `predict.py` — loads the trained model and predicts on new, held-out
  measurements; exposes `predict_quality()` for use by `app.py`.
- `app.py` — Streamlit demo (`streamlit run app.py`) for live prediction at
  your poster/oral presentation.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py              # train + evaluate on synthetic (or real) data
streamlit run app.py        # interactive demo
python predict.py           # quick command-line prediction check
python weight_analysis.py   # separate weight-loss summary
```

## Switching to real data

1. In `config.py`, set `USE_SYNTHETIC_DATA = False` and point
   `REAL_DATA_PATH` at your CSV.
2. Your CSV can be exactly as messy as your first real batch — "None",
   "10 g", "32%" etc. are all cleaned automatically by `data.py`. A
   `colour_value` outside 1-7 is clipped with a printed warning (check
   whether that row should actually be "decayed" instead).
3. For weight loss, point `WEIGHT_DATA_PATH` at a CSV with columns
   `banana_id, initial_weight_g, final_weight_g` and run
   `python weight_analysis.py` for the per-treatment summary (Section 3.10)
   — this stays separate from the day-by-day classifier.
4. Run `python main.py` again.

## Known data-quality flag from Batch 1

Your first real batch (B01/B02/B03) showed the 10g-dosage banana (B02) with
*higher* ethylene and *faster* spoilage than the untreated control — the
opposite of the hypothesis. Investigate that specific jar/sachet before
treating it as a normal data point (mislabeled sachet, seal failure, or
analyser fault are the likely causes) — the synthetic data generator in
`data.py` intentionally models the *intended* treatment effect, not this
anomaly, so don't expect it to reproduce that pattern.

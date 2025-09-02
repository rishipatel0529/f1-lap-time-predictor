# F1 Lap Time Predictor — Solo Edition

A lightweight, single‑machine project to predict **per‑lap time** from historical F1 telemetry, driver/team/circuit metadata, and simple lag features. Designed to run on a **MacBook Air** with no GPUs or heavy infra.

**Goal:** RMSE ≈ 4–6 seconds on held‑out race groups (reasonable for a compact baseline; actual results will vary by data quality and feature choices).

## Why this version?
- No Kafka/Feast/K8s/etc. — just **pandas + scikit‑learn**.
- Reproducible **GroupKFold** evaluation by race.
- Clean feature policy to avoid leakage (no sector‑split times, no per‑lap speeds measured within the lap).
- Saves artifacts: model (`models/model.joblib`), metrics (`artifacts/metrics.json`), and feature importance CSV.

## Folder layout
```
f1-lap-predictor-solo/
├─ src/
│  └─ f1lap/
│     ├─ featurize.py
│     ├─ train.py
│     ├─ utils.py
│     └─ __init__.py
├─ config/
│  └─ default.yaml
├─ models/
├─ artifacts/            # created on first run
├─ requirements.txt
├─ Makefile
└─ README.md
```

## Quickstart
1) Create and activate a virtual env (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Put your CSV at some path, e.g. `data/all_telemetry_track_data.csv`.  
(If you're using the attached example in ChatGPT, download it locally and update the path below.)

3) Train & evaluate with default config:
```bash
python -m src.f1lap.train --data data/all_telemetry_track_data.csv --config config/default.yaml
```

This will:
- Clean and featurize rows
- Build **lag features by driver per race**
- One‑hot encode drivers/teams/tyre compound/circuit
- Run **GroupKFold(n_splits=5)** grouped by `season_race`
- Train a **HistGradientBoostingRegressor** (fast, CPU‑friendly)
- Write artifacts under `artifacts/` and the trained model under `models/`

## Configuration
Edit `config/default.yaml` to tweak which columns to use/ignore, how many lags to add, and the evaluation setup.

## Tips to hit 4–6s RMSE
- **Filter pit‑in laps** or include `PitDuration` as a feature; pit laps are outliers.
- Add **rolling stats** (last 3/5 laps) for lap time & tyre life.
- Keep features that are known **before the lap begins** (avoid sector‑split times measured within the lap).
- Train by **race‑grouped splits** to simulate unseen sessions.
- Consider training **separate models per compound** or adding interactions (compound × tyre life × track temperature if available).

## Inference
After training, you can load `models/model.joblib` and call `predict(X)` where `X` is built using the same `featurize.py` logic on new data rows.

---

**Author**: You 🙂  
**License**: MIT

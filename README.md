# F1 Lap Time Predictor

> **Goal:** predict F1 race **lap_time** within ~4–6 seconds RMSE.
> **My result:** **RMSE ≈ 3.39s**, **MAE ≈ 1.38s** (from `artifacts/preds.csv`; values depend on your data/config).

This repository is a streamlined, single-developer version of a larger “F1 Race Strategy Optimization Platform.”
It focuses on the modeling loop you actually use: **feature engineering -> cross‑validated training -> batch prediction -> evaluation/plots**.  
Production systems (Kafka/Feast/Airflow/Kubernetes, RL online serving, canaries) are scoped out but documented for future growth.

The current model is a **scikit-learn HistGradientBoostingRegressor** trained on the merged CSV (`data/all_telemetry_track_data.csv`)
which already combines driver, event, tyre, track geometry and weather features. The pipeline supports safe feature
selection, simple lags, grouped CV, and writes small, git‑friendly artifacts.

---

## Table of Contents

- [Objectives & Success Metrics](#objectives--success-metrics)
- [What’s in This Repo](#whats-in-this-repo)
- [Quickstart](#quickstart)
- [Data Inputs & Schema](#data-inputs--schema)
  - [Column Reference (by section)](#column-reference-by-section)
    - [Core lap & session fields](#core-lap--session-fields)
    - [Categoricals & one-hots (normalized)](#categoricals--one-hots-normalized)
    - [Track & weather metadata](#track--weather-metadata)
    - [Corner Geometry Columns](#corner-geometry-columns)
- [Feature Engineering](#feature-engineering)
- [Model & Training](#model--training)
- [Evaluation & Artifacts](#evaluation--artifacts)
- [ETL Harness](#etl-harness)
- [Secrets & Configuration](#secrets--configuration)
- [Tech Stack Decisions](#tech-stack-decisions)
- [Data Sources](#data-sources)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Objectives & Success Metrics

These come from the original platform plan; this project primarily targets **Model Accuracy**.

### Business & Race Outcomes
- **Average lap-time reduction**: ≥ **0.3 s**  
- **Mean Time-To-Pit (MTTP) improvement**: ≥ **1.5 s**  
- **Pit-stop efficiency**: ≤ **2.5 s** end-to-end

### Model Accuracy & Learning
- **Pit-delta prediction error**: **MAE** ≤ **0.5s**  
- **RL policy reward uplift** vs. rule-based: ≥ **10%**

> **Delivered here:** lap-time prediction **RMSE ~3–4s** on the dataset.

### Performance & Scalability (platform targets)
- **End-to-end latency**: P95 ≤ **100ms**  
- **Inference throughput**: ≥ **2000rps**  
- **Feature store read latency**: P95 ≤ **30ms**  
- **Kafka ingestion throughput**: ≥ **5000eps**

### Reliability & Operations (platform targets)
- **Availability**: ≥ **99.98%**  
- **Message loss**: ≤ **0.5%**  
- **Full retraining**: ≤ **4hrs**  
- **Alert latency** (drift/errors): ≤ **60s**

### Quality & Maintainability
- **Unit-test coverage**: ≥ **85%** in `src/`  
- **CI green builds**: ≥ **95%** on `main`  
- **Docs completeness**: new modules ship with README/examples

---

## What’s in This Repo

- **Code**: `src/f1lap/` (feature build, training, predict CLI)
- **Configs**: `config/default.yaml` (features, lags, model), `config/etl.yaml` (optional ETL harness)
- **Scripts**: data collection files under `scripts/`
- **Artifacts** (git-kept & small):  
  - `artifacts/metrics_from_preds.json`  
  - `artifacts/slice_by_race.csv`, `slice_by_compound.csv`, `slice_by_stint.csv`  
  - `artifacts/residual_hist.png`
- **Model**: `models/model.joblib` (ignored by git; publish via Releases/LFS if needed)
- **Makefile**: `venv`, `etl`, `train`, `predict`, `etl-train`, `format`, `lint`
- **.gitignore**: excludes heavy data, caches, model binaries; keeps small summaries

---

## Quickstart

```bash
# 1) env
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 2) train on your merged CSV
python -m src.f1lap.train --data data/all_telemetry_track_data.csv --config config/default.yaml

# 3) batch predict
python -m src.f1lap.predict_cli --data data/all_telemetry_track_data.csv \
  --config config/default.yaml --model models/model.joblib --out artifacts/preds.csv

# 4) quick eval + plot (overwrites artifacts/residual_hist.png)
python - <<'PY'
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p = pd.read_csv("artifacts/preds.csv")
err = p["pred_lap_time"] - p["actual"]
rmse = float(np.sqrt((err**2).mean())); mae = float(err.abs().mean())
print(f"RMSE={rmse:.3f}s  MAE={mae:.3f}s  N={len(p)}")
plt.hist(err, bins=80, range=(-20, 20))
plt.title("Residuals (s)"); plt.xlabel("pred-actual (s)"); plt.ylabel("count")
plt.axvline(err.mean(), ls="--"); plt.axvline(np.median(err), ls=":")
plt.savefig("artifacts/residual_hist.png", dpi=160, bbox_inches="tight")
PY
```

---

## Data Inputs & Schema

Primary input: **`data/all_telemetry_track_data.csv`** (≈138k rows and 219 columns in the run).  

### Column Reference (by section)

#### 1) Core lap & session fields

| Field | Description |
|---|---|
| `season` | Championship year (e.g., 2019–2024). |
| `race_round` | Calendar round number within the season. |
| `driver_id` | Driver jersey number. |
| `lap_number` | Lap index for this record (1…N). |
| `position` | On-track position at lap completion. |
| `grid_position` | Starting grid position. |
| `finish_position` | Final race ranking (**excluded from training** to avoid leakage). |
| `lap_time` | Lap duration in seconds (**training target**). |
| `Stint` | Stint counter for the driver in this race. |
| `pit_stop_lap` | Lap on which a pit stop occurred (if any). |
| `PitDuration` | Pit service duration in seconds. |
| `FreshTyre` | Whether tyres were fresh on this lap (bool). |
| `TyreLife` | Laps accumulated on current tyres before this lap. |
| `IsPersonalBest` | Whether this was driver’s PB at the time (bool). |
| `SpeedFL` | Speed at the **Finish Line** (km/h). |
| `SpeedI1` | Speed at **Sector 1** timing line (km/h). |
| `SpeedI2` | Speed at **Sector 2** timing line (km/h). |
| `SpeedST` | **Speed Trap** reading (km/h). |
| `Time` | Session clock (s) / elapsed time for the session. |
| `LapStartDayOfYear` | Day-of-year of lap start (1–366). |
| `LapStart_sin`, `LapStart_cos` | Cyclical encoding of lap start time. |
| `sector1_time` | S1 time (s). |
| `sector2_time` | S2 time (s). |
| `sector3_time` | S3 time (s). |
| `Sector1SessionTime` | Session clock at S1 crossing (s). |
| `Sector2SessionTime` | Session clock at S2 crossing (s). |
| `Sector3SessionTime` | Session clock at S3/finish (s). |
| `humidity` | Ambient humidity (%) near lap time. |
| `Deleted` | Lap deleted (track limits/flags) indicator. |
| `FastF1Generated` | Row derived/filled by FastF1 pipeline (bool). |

> In `config/default.yaml`, we exclude leakage-prone columns by default.

---

### 2) Categoricals & one-hots (normalized)

| Field | Description |
|---|---|
| `Compound_type` | Normalized tyre compound (`SOFT/MEDIUM/HARD/INTERMEDIATE/WET/UNKNOWN`). |
| `Driver_name` | Driver code/name (normalized alternative to `Driver_*` one-hots). |
| `Team_name` | Team name (normalized alternative to `Team_*` one-hots). |
| `grand_prix_(track_name)` | Event/track name (normalized alternative to `grand_prix_*` one-hots). |
| `Compound_*` | One-hot flags for compounds (e.g., `Compound_SOFT`). |
| `Driver_*` | One-hot flags for drivers (e.g., `Driver_NOR`, `Driver_VER`, …). |
| `Team_*` | One-hot flags for teams (e.g., `Team_McLaren`, `Team_Mercedes`, …). |
| `grand_prix_*` | One-hot flags for GP events (e.g., `grand_prix_bahrain_grand_prix`). |

**Drivers present (values):**  
`AIT, ALB, ALO, BEA, BOT, COL, DEV, DOO, FIT, GAS, GIO, GRO, HAM, HUL, KUB, KVY, LAT, LAW, LEC, MAG, MAZ, MSC, NOR, OCO, PER, PIA, RAI, RIC, RUS, SAI, SAR, STR, TSU, VER, VET, ZHO`

**Grand Prix events covered:**  
**70th Anniversary (Silverstone 2020), Abu Dhabi, Australian, Austrian, Azerbaijan, Bahrain, Belgian, Brazilian, British, Canadian, Chinese, Dutch, Eifel, Emilia-Romagna, French, German, Hungarian, Italian, Japanese, Las Vegas, Mexico City, Miami, Monaco, Portuguese, Qatar, Russian, Sakhir (Bahrain 2020), São Paulo, Saudi Arabian, Singapore, Spanish, Styrian (Austrian 2020–2021), Turkish, Tuscan, United States**.

> One-hot columns are auto-included via `onehot_prefixes`.

---

### 3) Track & weather metadata

| Field | Description |
|---|---|
| `track_length` | Circuit length (km). |
| `number_of_laps` | Scheduled race distance in laps. |
| `corners` | Number of corners in the layout. |
| `race_direction` | `Clockwise` / `Anti-Clockwise`. |
| `pit_lane_length` | Pit lane length (m). |
| `current_time(GMT-0)` | UTC timestamp aligned to lap start (if available). |
| `weather_condition` | Text condition near lap (e.g., *Sunny, Light rain*). |
| `current_temperature(°C)` | Ambient/track temperature (°C) near lap. |

---

### 4) Corner geometry columns

| Field pattern | Description |
|---|---|
| `corner_{i}_radius` | Corner radius (m). |
| `corner_{i}_distance` | Distance from previous reference to this corner (m). |
| `corner_{i}_elev_change` | Elevation change through the corner (m, signed). |
| `corner_{i}_entry_speed` | Typical entry speed (km/h). |
| `corner_{i}_exit_speed` | Typical exit speed (km/h). |
| `corner_{i}_max_speed` | Peak speed within the corner (km/h). |

---

## Feature Engineering

- **Safe candidates** (see `config/default.yaml`):
  - `candidate_numeric`: pre-lap variables (grid_position, TyreLife, humidity, LapStart encodings, track basics, etc.)
  - `candidate_categorical`: `race_direction`, `grand_prix`, `compound`, `driver_id`, `Team`
  - Auto-one-hots from prefixes: `Driver_*`, `Team_*`, `grand_prix_*`, `Compound_*`
- **Exclusions to prevent leakage**: raw speed traps, finish_position
- **Lags** per `(season, race_round, driver_id)`:  
  - `lap_time`: lags `1, 2, 3`  
  - `TyreLife`: lags `1, 2`
- **Filtering**: drop laps with `lap_time < 40 s` or `> 300 s`; optional drop pit-in laps (`PitDuration > 0`)

---

## Model & Training

- **Estimator**: `HistGradientBoostingRegressor` (CPU-friendly, robust on tabular data)  
  - defaults: `max_iter=400`, `learning_rate=0.08`, `min_samples_leaf=20`, `max_depth=None`
- **Grouped CV**: 5-fold grouped by `season,race_round` to avoid same-race leakage
- **Residualized target**: if the saved model bundle includes a `baseline_map`, learn `lap_time - baseline` and add it back at inference; the default training here uses direct `lap_time`.
- **Bundle**: `models/model.joblib` contains the model + feature names + simple residual quantiles for uncertainty bands

---

## Evaluation & Artifacts

After `predict_cli`, `artifacts/preds.csv` contains:  
`pred_lap_time, pred_low, pred_high, actual, error`.

Example headline metrics from your run:
- **Overall**: `RMSE ≈ 3.391s`, `MAE ≈ 1.383s`, coverage(P10–P90) ≈ **86%**
- **By compound (sample)**: HARD ≈ **3.26s**, MEDIUM ≈ **3.24s**, SOFT ≈ **3.90s**, INTERMEDIATE/WET higher (expected)

Saved artifacts (small, tracked):
- `artifacts/metrics_from_preds.json`
- `artifacts/slice_by_race.csv`, `artifacts/slice_by_compound.csv`, `artifacts/slice_by_stint.csv`
- `artifacts/residual_hist.png` with mean/median markers


> Large outputs (`artifacts/preds.csv`), raw `data/**`, and `models/**` are git-ignored.

---

## ETL Harness

Run:

```bash
python -m src.f1lap.build_dataset_cli --config config/etl.yaml
```

This will **attempt** to run your legacy scripts (FastF1/Ergast/weather/track geometry) listed in `config/etl.yaml`.  
Missing scripts are skipped; present scripts may require `.env` keys. Output is written to:

```
data/all_telemetry_track_data.csv
```

You can skip ETL entirely if you already have that CSV.

---

## Secrets & Configuration

Put API keys in **`config/.env`** (ignored by git):

```
OWM_API_KEY=                  # used to fetch current/forecast weather
WANDB_API=                    # experiment tracking for params/metrics/artifacts (optional)
RAPIDAPI_KEY=                 # source for alternate data sources (e.g., backup weather or motorsport endpoints) (used to check data) (optional)
SPORTSRADAR_API_KEY=          # enriching event calendars and validating race rounds
F1TV_USERNAME=                # only needed if you run the F1TV fetch script to pull (paid subscription)
F1TV_PASSWORD=                # password for F1_TV
FASTF1_CACHE_PATH=            # data/track_data_csv_files (the path I used)
VISUALCROSSING_API_KEY=       # to retrieve historical or hourly weather aligned to events/laps
GOOGLE_ELEVATION_API_KEY=     # elevation deltas (elev_change) for track geometry features
```

Load via `python-dotenv` or your own script logic.

---

## Tech Stack Decisions

- **Kafka (self-managed vs Confluent Cloud)**: control vs zero-ops trade-off  
- **Feast OSS vs SageMaker Feature Store**: open & flexible vs managed AWS integration  
- **Airflow vs Prefect**: mature ecosystem vs simpler Pythonic API  
- **Minikube vs EKS**: local/dev vs managed production Kubernetes

**Project selection:** local workflow; production components can be layered in later.

---

## Data Sources

- **Live telemetry** via Kafka topics (packets, driver state, race events)  
- **Weather** via OpenWeatherMap One Call (3.0) with `OWM_API_KEY` and visualcrossing with `VISUALCROSSING_API_KEY`
- **Historical** via Ergast + FastF1 (2018+), stored as Parquet

*This project build trains directly from your merged CSV.*

---

## Reproducibility
- **Environment**: `requirements.txt` (lean core)
- **Makefile**: `venv`, `etl`, `train`, `predict` standardize runs
- **Git hygiene**: large data/models ignored; small summaries tracked

---

## License

**MIT** — see [LICENSE](./LICENSE).

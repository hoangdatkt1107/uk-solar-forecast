# GridSight UK

**Probabilistic solar generation forecasting for the GB national grid** — half-hourly,
6–12 hours ahead, as a p10 / p50 / p90 range rather than a single number.

**It cuts the grid operator's own forecast error by about a third on genuinely out-of-sample data — and it is live.**

<p>
  <a href="https://orange-mushroom-083511803.7.azurestaticapps.net"><b>▶ Live dashboard</b></a>
  · a personal project by <b>Tien Dat Hoang</b>
</p>

![live](https://img.shields.io/badge/status-live%20on%20Azure-success)
![serve](https://img.shields.io/badge/serve-hourly%20·%202%20vCPU%20%2F%204%20GiB-blue)
![retrain](https://img.shields.io/badge/retrain-weekly%20·%20GitHub%20Actions-black)
![python](https://img.shields.io/badge/python-3.13-blue)
![license](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)

---

## Results

The benchmark is **NESO** — the National Energy System Operator's own embedded-solar
forecast, i.e. the number GB's grid is actually balanced on. Beating it is the whole point.

Two years of backtest are on the dashboard, but the deployed model is **retrained weekly on a
rolling split**, so most of that span is data it has already trained on. The honest test is the
**genuinely out-of-sample** stretch — the months after the model's training cut-off (2026-01
onward, ~4,800 daylight slots). Scored on **daylight slots only** (night is trivially zero and
would flatter every model):

| Horizon | GridSight MAE | NESO MAE | Skill (MAE reduction) | p10–p90 coverage |
|---|---|---|---|---|
| **6 h ahead** | **703 MW** | 1080 MW | **−34.9 %** | 80.4 % |
| **12 h ahead** | **716 MW** | 1080 MW | **−33.7 %** | 80.5 % |

<sub>Over the full 2-year record the same model scores ~−55 % skill at ~93 % coverage — but that
span is largely **in-sample** (the weekly-retrained model had already trained on it), so those
headline-friendly figures are optimistic. The dashboard shades the in-sample region and reports
the out-of-sample numbers instead; this README does the same. Getting a true 2-year out-of-sample
curve would need a walk-forward backtest (each period predicted by a model trained only on prior
data) — a known open item.</sub>

- **About a third off the operator's error on unseen data**, and the margin holds across seasons
  — scrub the timeline on the dashboard rather than taking one good week's word for it.
- **The intervals are well-calibrated out-of-sample.** ~80 % of actuals land inside the
  nominally-80 % p10–p90 band — close to target. (The wider ~93 % seen over the full record was
  an in-sample artefact, not genuine conservatism.)
- **The foundation model lost.** A univariate zero-shot **Chronos** baseline stays *worse than
  NESO* (negative skill) even out-of-sample. Weather (NWP irradiance) and the physics of solar
  geometry carry this problem; a general-purpose time-series model that never sees the sky
  cannot compete. That negative result is kept in the repo and on the dashboard on purpose.

**How the forecast is built:** a quantile stack — **LightGBM** (tabular, weather + calendar
+ lags) and a **TCN** (sequence) each predict q10/q50/q90, and a linear quantile
meta-learner combines them out-of-fold, with a clear-sky physical term as a feature.

**Contents** · [Architecture](#architecture) · [Why it is deployed this way](#why-it-is-deployed-this-way)
· [Repository Structure](#repository-structure) · [Quick Start](#quick-start)
· [Data Pipeline](#data-pipeline--bronze--silver--gold) · [Feature Reference](#gold-feature-reference)
· [Environment Setup](#environment-setup)

---

## Architecture

The system splits into **three jobs on different schedules**, deliberately placed on
different infrastructure (see [why](#why-it-is-deployed-this-way)).

```
  DATA SOURCES (live, free)                    ┌──────────────────────────────┐
 ┌─────────────────────────┐                   │      HuggingFace Hub         │
 │ Met Office UK-2km NWP   │  new init /3h     │   (the data lake + registry) │
 │   (AWS S3 open data)    │────┐              │                              │
 │ PV_Live GSP actuals     │    │              │  gridsight-bronze  (parquet) │
 │   (api.pvlive.uk) /30m  │────┼─────────────▶│  gridsight-silver / -gold    │
 │ NESO embedded forecast  │    │   ingest +   │  gridsight-model  (registry) │
 │   (CKAN API) /1h        │────┘   push back  └──────────────────────────────┘
 └─────────────────────────┘                      ▲          │            ▲
                                                  │          │ pull model │ push
                                        push bronze          ▼            │ promoted
                                                  │   ┌──────────────┐    │ model
   ╔══════════════ AZURE ═══════════════╗         │   │              │    │
   ║                                    ║         │   │              │    │
   ║  Container Apps JOB  (cron 0 * * *)║─────────┘   │              │    │
   ║  gridsight-serve · 2 vCPU / 4 GiB  ║◀────────────┘              │    │
   ║   refresh tail → silver → gold     ║                            │    │
   ║   → forecast → forecast_*.json     ║                            │    │
   ║          │ writes                  ║              ╔═════════ GITHUB ACTIONS ══════╗
   ║     ┌────▼─────┐                   ║              ║                               ║
   ║     │Azure File│  share "serve"    ║              ║  weekly-retrain (cron Sun 3am)║
   ║     └────┬─────┘                   ║              ║  free runner · 16 GB RAM      ║
   ║          │ reads                   ║              ║   full history → gold → train ║
   ║  ┌───────▼────────┐                ║              ║   → promote model             ╫
   ║  │ Container App  │ FastAPI, 24/7  ║              ║                               ║
   ║  │ gridsight-api  │ 0.25 vCPU      ║              ╚═══════════════════════════════╝
   ║  └───────┬────────┘                ║
   ║          │ /forecast (CORS)        ║
   ║  ┌───────▼────────┐                ║
   ║  │ Static Web App │  dashboard     ║
   ║  │ (Free tier)    │  + history.json║
   ║  └────────────────┘                ║
   ╚════════════════════════════════════╝
```

**The three jobs**

| Job | Where | When | Weight | What it does |
|---|---|---|---|---|
| **serve** | Azure Container Apps Job | hourly (`0 * * * *`) | ~3 min, <4 GiB | Pull recent tail from the sources, push fresh bronze back to HF, rebuild silver+gold, **pull the live model from HF**, write `forecast_*.json` to Azure Files |
| **api** | Azure Container App | always on | 0.25 vCPU | Read the JSON and serve `/forecast` — never touches the model |
| **retrain** | **GitHub Actions** | weekly (Sun 03:00 UTC) | ~5.5 GiB peak | Pull the **full** history, rebuild gold, train both horizons on a rolling split, **push the promoted model to HF** |

The two compute jobs never talk to each other — **HuggingFace is the handoff**. Retrain
publishes a model; serve picks it up on its next hourly run. No redeploy, no coupling.

### Why it is deployed this way

This is a personal project on **free / near-free tiers**, and the shape follows directly
from that budget:

- **Data lake on HuggingFace, not Azure Storage.** HF hosts the parquet lake for free and
  `snapshot_download` is incremental. Paying for ADLS would buy nothing here. Bronze must be
  pushed back **every run** because Met Office S3 only keeps a rolling window of recent
  inits — NWP not saved is gone for good.
- **Serve on Azure, sized to fit 4 GiB.** The Container Apps environment is a *legacy
  Consumption* one, hard-capped at **2 vCPU / 4 GiB** (8 GiB would need a different
  environment type and a new API URL). So the hourly job is tuned to fit: stack-only (no
  Chronos), a 2-month bronze window, and current-year NESO. It runs in ~3 min, billed per
  execution-second — a few $/month.
- **Retrain on GitHub Actions, not Azure.** Training peaks at **~5.5 GiB** — over the 4 GiB
  cap. Rather than pay to upgrade the whole environment for one job a week, the weekly
  retrain runs on GitHub's free 16 GB runner and hands the model over via HF. Free, and the
  showcase (live hourly serving) still runs on Azure.
- **Dashboard as a Static Web App (Free).** One HTML page reading one API; a framework and
  a server would be cost with no benefit. The 2-year backtest ships as a static
  `history.json`; only the next-12h overlay is fetched live.
- **Scale-to-zero everywhere it is possible.** The serve job only exists while it runs; the
  API is the one always-on piece and is deliberately tiny because it never loads a model.

---

## Repository Structure

```
gridsight-uk-2026/
│
├── data_ingestion/              # Bronze → Silver → Gold data pipeline
│   ├── bronze/                  # Raw ingestion (NESO, PV_Live, OCF, Met Office AWS) → parquet + HF upload
│   ├── silver/                  # Clean / align / quality-check each source (neso.py builds per-file, memory-safe)
│   ├── gold/                    # Feature engineering → model-ready table (merge, targets, calendar, lags)
│   └── sync_bronze.py           # Pull Bronze from HF → local data/bronze/
│
├── modeling/                    # Models, training, live serving
│   ├── config.py                # ModelConfig — rolling val/test split, horizons, hyperparams
│   ├── data.py                  # Gold loader + split_masks (rolling or pinned dates)
│   ├── train.py                 # TCN-Q + LGBM-Q + Linear-Q stack training
│   ├── serve.py / predict.py    # Build gold + forecast the next horizon
│   ├── registry.py              # HF model registry — pull live model / promote a retrain
│   └── evaluate.py · metrics.py · clearsky.py · stacking.py · chronos_baseline.py · cli.py
│
├── serving/
│   └── api.py                   # FastAPI — reads forecast JSON, serves /forecast + /health
│
├── frontend/                    # Vanilla dashboard (Azure Static Web Apps)
│   ├── index.html               # History explorer + live next-12h overlay
│   ├── build_history.py         # Generates history.json (2-year backtest)
│   └── staticwebapp.config.json
│
├── deploy/                      # Azure Container Apps deploy — see deploy/README.md
│   ├── bicep/main.bicep         # API + serve job + Key Vault + storage + managed identity
│   ├── Dockerfile.api · Dockerfile.job
│   └── scripts/                 # deploy.sh, run_serve_now.sh, deploy_frontend.sh, config.env
│
├── .github/workflows/
│   └── retrain.yml              # Weekly retrain on GitHub Actions → push model to HF
│
├── pipeline.py                  # Cloud entrypoint: `python pipeline.py serve | retrain`
├── src/gridsight/               # Shared settings/config
├── configs/
├── requirements.txt · requirements-api.txt · requirements-job.txt
├── .env                         # HF_TOKEN (git-ignored)
└── README.md · deploy/README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- A HuggingFace token with read access to the dataset repos
- `HF_TOKEN` set in `.env` (see [Environment Setup](#environment-setup))

### Full local build

```bash
# 1. Pull Bronze data from HuggingFace
./env/bin/python -m data_ingestion.sync_bronze --source all

# 2. Build Silver from local Bronze (no network)
./env/bin/python -m data_ingestion.silver --source all

# 3. Build Gold for the two horizons the live models use (no network)
./env/bin/python -m data_ingestion.gold --horizon-steps 24   # 12 h ahead (primary)
./env/bin/python -m data_ingestion.gold --horizon-steps 12   # 6 h ahead
```

Gold is written per horizon to `data/gold/gold_features_{12h,6h}/year=YYYY/month=MM/` — the
tables are separate because the leakage-safe lags depend on the horizon.

> The CLI's bare default is `--horizon-steps 48` (24 h day-ahead), a leftover from an
> earlier scope. The deployed system runs `GRIDSIGHT_HORIZONS=24,12` — pass the flag
> explicitly to reproduce what is live.

### Skip rebuilding — pull pre-built data directly

If you only need the data and do not want to recompute it, pull Silver and Gold directly from the HuggingFace repos:

```bash
# Log in once (needed for private repos)
./env/bin/hf auth login
# or append --token <your_hf_token> to each command below

# Pull Silver
./env/bin/hf download Masonhoang1107/gridsight-silver \
    --repo-type dataset --local-dir data/silver

# Pull Gold
./env/bin/hf download Masonhoang1107/gridsight-gold \
    --repo-type dataset --local-dir data/gold
```

---

## Data Pipeline — Bronze → Silver → Gold

The pipeline follows the **medallion architecture**: raw → clean → model-ready. Each layer reads only the previous layer and is rebuildable from scratch.

```
Internet APIs / HuggingFace
          │
          ▼
    ┌──────────┐
    │  BRONZE  │  Raw parquet, immutable, partitioned year=/month=
    └──────────┘
          │  (local only, no network)
          ▼
    ┌──────────┐
    │  SILVER  │  UTC-aligned, validated, 30-min grid, quality-flagged
    └──────────┘
          │  (local only, no network)
          ▼
    ┌──────────┐
    │   GOLD   │  Leakage-safe feature table, ready for model training
    └──────────┘
```

---

### Bronze — Raw Downloads

Bronze does one job: download the raw data and save it to parquet. No cleaning, no joining. Each source gets its own folder, partitioned by `year=YYYY/month=MM/`.

#### Ingest all sources for specific years

```bash
./env/bin/python -m bronze --source all --years 2021 2022 2023 2024
```

#### Ingest a single source

```bash
./env/bin/python -m bronze --source pv_live      --years 2021 2022 2023 2024
./env/bin/python -m bronze --source neso         --years 2021 2022 2023 2024
./env/bin/python -m bronze --source ocf_pv       --years 2021 2022 2023 2024
./env/bin/python -m bronze --source met_office_nwp --years 2021 2022 2023 2024 --hours 0 12
```

#### Met Office NWP — additional options

```bash
# Only 00Z and 12Z init-times (recommended — covers day-ahead horizon)
./env/bin/python -m bronze --source met_office_nwp --years 2023 2024 --hours 0 12

# All 24 init-times per day (large — use for research only)
./env/bin/python -m bronze --source met_office_nwp --years 2023 2024 --hours -1

# Parallel workers (default 4)
./env/bin/python -m bronze --source met_office_nwp --years 2023 2024 --workers 8

# Re-extract files that already exist
./env/bin/python -m bronze --source met_office_nwp --years 2023 2024 --overwrite
```

#### NESO — additional options

```bash
# Specific CKAN package IDs
./env/bin/python -m bronze --source neso --packages embedded-wind-and-solar-forecasts

# Force re-download even if last_modified is unchanged
./env/bin/python -m bronze --source neso --force-refresh

# Cap rows per resource (for dev/testing)
./env/bin/python -m bronze --source neso --max-records 10000
```

#### Upload Bronze to the HF repo after ingestion

```bash
./env/bin/python -m bronze --source all --years 2021 2022 2023 2024 --upload
```

#### Output structure

```
data/bronze/
├── pv_live/
│   └── year=2024/month=05/gsp_observations.parquet
├── neso/
│   └── embedded-wind-and-solar-forecasts/
│       ├── embedded_wind_and_solar_forecasts_2024.parquet
│       └── _state.json                   # incremental cache
├── ocf_pv/
│   └── year=2024/month=05/<uuid>.parquet  # multiple files per month
└── met_office_nwp/
    └── year=2024/month=05/nwp_20240501-00Z.parquet
```

---

### Silver — Clean & Align

Silver reads local Bronze and applies four deterministic cleaning rules to every source, then validates the result with hard-fail contracts. All Silver tables share a canonical UTC 30-minute time index (`timestamp_utc`).

#### Build all Silver tables

```bash
./env/bin/python -m data_ingestion.silver --source all
```

#### Build a single table

```bash
./env/bin/python -m data_ingestion.silver --source pv_live
./env/bin/python -m data_ingestion.silver --source met_office_nwp
./env/bin/python -m data_ingestion.silver --source ocf_pv
./env/bin/python -m data_ingestion.silver --source neso
```

#### Cross-source sanity checks

```bash
./env/bin/python -m data_ingestion.silver --source cross_check
```

Checks that the correlation between `pv_live.generation_mw` and `neso.embedded_solar_mw` is > 0.85 — a sanity check that both sources are measuring the same physical quantity.

#### Push Silver to the HF repo

```bash
./env/bin/python -c "from data_ingestion.silver import upload_silver_to_hf; upload_silver_to_hf()"
```

#### Cleaning rules applied to every source

| Rule | What it does |
|---|---|
| **UTC alignment** | All timestamps converted to tz-aware UTC, floored to the nearest 30-min slot |
| **Range clamping** | Physics-impossible values (e.g. generation > 15,000 MW) are set to NaN |
| **Canonical reindex** | Each table is reindexed onto an unbroken 30-min UTC grid; missing slots become explicit NaN rows (never silently dropped) |
| **Missing-data policy** | Gaps ≤ 1 step → forward-filled, flagged `ffill`; gaps 2–6 steps → NaN, flagged `gap`; gaps 7+ steps → NaN, flagged `long_gap` |

#### Source-specific notes

| Source | Key logic |
|---|---|
| **pv_live** | PV_Live uses period-end timestamps; shifted −30 min to period-start before flooring. Duplicate fetches for the same slot resolved by keeping the freshest. |
| **met_office_nwp** | 7 UK regional point forecasts weighted by regional PV capacity → single national weighted average per variable. `nwp_age_h` column added. Only the latest available init-time is kept per valid-time slot. Anti-leakage: rows where `init_time > valid_time` are dropped. |
| **ocf_pv** | Individual site `generation_Wh` values summed per 30-min slot to produce `ocf_total_mw` (÷ 500,000 to convert Wh → MW). Processed one Bronze file at a time to stay memory-efficient. |
| **neso** | Archive files (with issue timestamps) preferred over rolling live-forecast files. Anti-leakage: rows where `Forecast_Datetime > slot_timestamp` are dropped. |

#### Output structure

```
data/silver/
├── silver_pv_live/year=2024/month=05/silver_pv_live_202405.parquet
├── silver_met_office_nwp/year=2024/month=05/silver_met_office_nwp_202405.parquet
├── silver_ocf_pv/year=2024/month=05/silver_ocf_pv_202405.parquet
└── silver_neso/year=2024/month=05/silver_neso_202405.parquet
```

---

### Gold — Feature Store

Gold reads local Silver and produces the final feature table. The entire build is parameterised by `horizon` (number of 30-min steps ahead). This controls which lag features are leakage-safe — all lags are forced to be ≥ horizon.

#### Build for day-ahead forecasting (horizon = 48 steps = 24h) — default

```bash
./env/bin/python -m data_ingestion.gold
```

#### Build for 3-hour-ahead forecasting (horizon = 6 steps)

```bash
./env/bin/python -m data_ingestion.gold --horizon-steps 6
```

#### Build and push to the HF repo in one command

```bash
./env/bin/python -m data_ingestion.gold --upload
```

#### Push Gold to the HF repo separately

```bash
./env/bin/python -m data_ingestion.gold --upload
```

#### Build steps

```
merge_silver()         ← LEFT JOIN 4 Silver tables on timestamp_utc (pv_live is the spine)
    │
add_targets()          ← target_mw, target_cf
    │
add_calendar()         ← hour, half_hour, dow, month, doy, is_weekend, tod_sin/cos, doy_sin/cos
    │
add_solar()            ← solar_elevation_deg, clearsky_cos, is_daylight  (NOAA formula)
    │
add_lags_rolling()     ← gen_lag_N, cf_lag_N, gen_roll_mean_N, gen_roll_std_N  (all ≥ horizon)
    │
drop leaky columns     ← generation_mw, ocf_total_mw, ocf_mean_wh, ocf_n_systems removed
    │
validate_gold()        ← hard-fail anti-leakage contract checks
    │
write_gold()           ← data/gold/gold_features/year=YYYY/month=MM/
```

#### Anti-leakage contract checks (crash the build if violated)

- Raw observed columns (`generation_mw`, `ocf_total_mw`) must NOT appear as features — they would be direct target leakage
- Every lag column must use a lag ≥ horizon — `gen_lag_10` in a `horizon=48` build means the model sees data from 5 hours ago when predicting 24 hours ahead
- Night slots (solar elevation < −5°) must have near-zero generation (physical sanity)
- `timestamp_utc` must be unique, monotonically increasing, and on a 30-min grid

#### Output structure

```
data/gold/
└── gold_features/
    └── year=2024/month=05/gold_features_202405.parquet
```

---

## Gold Feature Reference

The Gold table (`gold_features`) has **47 columns** on a 30-minute UTC grid (`timestamp_utc`). One row per half-hour slot.

### Targets

| Column | Type | Description |
|---|---|---|
| `target_mw` | float32 | National PV generation at t (MW) — **primary target** |
| `target_cf` | float32 | Capacity factor = generation / capacity_mwp, clipped [0, 1.5] — capacity-normalised target |
| `capacity_mwp` | float32 | Installed PV capacity at t (MWp) — slow-moving, safe to use as feature |

### Weather — Met Office NWP (forecast, known ahead of t)

| Column | Type | Description |
|---|---|---|
| `ssrd_uk` | float32 | Surface downwelling shortwave radiation, UK weighted (W/m²) — **top solar driver** |
| `tcc_uk` | float32 | Total cloud cover, UK weighted (0–1) |
| `lcc_uk` | float32 | Low cloud cover, UK weighted (0–1) |
| `t2m_uk` | float32 | 2 m air temperature, UK weighted (K) |
| `ws10_uk` | float32 | 10 m wind speed, UK weighted (m/s) |
| `nwp_age_h` | float32 | Forecast age in hours (0–15; NWP source caps at 15 h) |

### Operator baseline — NESO (forecast, known ahead of t)

| Column | Type | Description |
|---|---|---|
| `embedded_solar_mw` | float32 | NESO embedded solar forecast (MW) — strong feature; NESO MAE ≈ 317 MW |
| `embedded_wind_mw` | float32 | NESO embedded wind forecast (MW) |
| `embedded_solar_capacity_mw` | float32 | NESO solar capacity context (MW) |
| `embedded_wind_capacity_mw` | float32 | NESO wind capacity context (MW) |

### Calendar (deterministic — always known for any future t)

| Column | Type | Description |
|---|---|---|
| `hour` | int16 | Hour of day (0–23) |
| `half_hour` | int16 | Half-hour index (0–47) |
| `dow` | int16 | Day of week (0 = Monday) |
| `month` | int16 | Month (1–12) |
| `doy` | int16 | Day of year (1–366) |
| `is_weekend` | int8 | 1 if Saturday or Sunday |
| `tod_sin`, `tod_cos` | float32 | Cyclical time-of-day encoding (eliminates midnight discontinuity) |
| `doy_sin`, `doy_cos` | float32 | Cyclical day-of-year encoding (eliminates year-end discontinuity) |

### Solar geometry (deterministic — NOAA formula at UK centroid 54°N, −2.5°W)

| Column | Type | Description |
|---|---|---|
| `solar_elevation_deg` | float32 | Sun elevation angle at UK centroid (degrees) |
| `clearsky_cos` | float32 | cos(zenith) clipped ≥ 0 — theoretical insolation factor (0 at night) |
| `is_daylight` | int8 | 1 if sun above horizon |

### Lagged / rolling — observed actuals (leakage-safe, all shifted ≥ horizon)

For `horizon=48` (day-ahead), all lags are ≥ 48 steps (≥ 24 hours ago).

| Column | Type | Description |
|---|---|---|
| `gen_lag_48` | float32 | Generation 24h ago (MW) |
| `gen_lag_96` | float32 | Generation 48h ago (MW) |
| `gen_lag_144` | float32 | Generation 72h ago (MW) |
| `gen_lag_336` | float32 | Generation 7 days ago (MW) |
| `cf_lag_{48,96,144,336}` | float32 | Capacity factor at the same lags |
| `gen_roll_mean_48` | float32 | Trailing 24h mean generation (window ends at t − horizon) |
| `gen_roll_mean_336` | float32 | Trailing 7-day mean generation |
| `gen_roll_std_48` | float32 | Trailing 24h generation volatility |
| `cf_roll_mean_48` | float32 | Trailing 24h mean capacity factor |
| `cf_roll_mean_336` | float32 | Trailing 7-day mean capacity factor |
| `ocf_lag_48` | float32 | OCF rooftop-fleet index 24h ago (MW) |
| `ocf_roll_mean_48` | float32 | Trailing 24h mean OCF fleet index |

### Quality / bookkeeping (not model inputs)

| Column | Type | Description |
|---|---|---|
| `pv_flag` | string | Per-slot PV_Live quality flag: `ok` / `ffill` / `gap` / `long_gap` |
| `nwp_flag` | string | Per-slot NWP quality flag |
| `neso_flag` | string | Per-slot NESO quality flag |
| `ocf_flag` | string | Per-slot OCF quality flag |
| `has_full_history` | int8 | 0 for warmup rows that lack enough history for the longest lag/rolling feature; 1 otherwise |

---

## HuggingFace Repositories

| Layer | HF Dataset Repo | Purpose |
|---|---|---|
| Bronze | `Masonhoang1107/gridsight-bronze` | Raw immutable downloads |
| Silver | `Masonhoang1107/gridsight-silver` | Cleaned, validated, UTC-aligned tables |
| Gold | `Masonhoang1107/gridsight-gold` | Model-ready feature store |

### Sync Bronze from HuggingFace (recommended first step)

```bash
# All sources
./env/bin/python -m data_ingestion.sync_bronze --source all

# Single source
./env/bin/python -m data_ingestion.sync_bronze --source met_office_nwp
```

The sync is incremental and resumable — files already present locally are skipped.

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/hoangdatkt1107/GridSight_UK.git
cd GridSight_UK
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv env
source env/bin/activate         # macOS / Linux
# env\Scripts\activate.bat      # Windows

pip install -r requirements.txt
```

### 3. Set your HuggingFace token

Create a `.env` file in the project root (already git-ignored):

```bash
# .env
HF_TOKEN="hf_your_token_here"
```

You can generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). The token needs **read** access to the `Masonhoang1107` dataset repos.

### 4. Verify installation

```bash
python -m data_ingestion.sync_bronze --help
python -m data_ingestion.silver --help
python -m data_ingestion.gold --help
```

### Dependencies

Pinned in three requirement sets, split so each container installs only what it needs:

| File | Used by | Contains |
|---|---|---|
| `requirements-job.txt` | serve + retrain job, GitHub Actions retrain | Everything: ingest (`httpx`, `huggingface_hub`), NWP extraction (`xarray`, `netCDF4`, `pyproj`), dataframes (`pandas`, `pyarrow`), modeling (`lightgbm`, `scikit-learn`, `scipy`, `joblib`), Chronos (`chronos-forecasting`, `transformers`). **`torch` is installed separately** from the CPU wheel index — see `deploy/Dockerfile.job`. |
| `requirements-api.txt` | FastAPI container | Minimal — the API only reads JSON, so it never installs torch or LightGBM. That is why it runs in 0.25 vCPU. |
| `requirements.txt` | local / notebooks | Base data-pipeline set. |

```bash
# local dev — the full job set (matches what runs in the cloud)
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.12.1"
pip install -r requirements-job.txt
```

### Models in this repo

| Component | File | Role |
|---|---|---|
| **LGBM-Q** | `modeling/base/lgbm_q.py` | Gradient-boosted quantile regressor over the tabular features |
| **TCN-Q** | `modeling/base/tcn_q.py` | Temporal convolutional net over the 126-step sequence |
| **Linear-Q stack** | `modeling/stacking.py` | Meta-learner combining both, fit on out-of-fold predictions |
| **Chronos** | `modeling/chronos_baseline.py` | Univariate zero-shot baseline (`chronos-bolt-base`) — kept for comparison; it loses to NESO |

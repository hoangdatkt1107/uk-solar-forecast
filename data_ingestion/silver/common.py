from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gridsight.config import settings

BRONZE_LOCAL_DIR = settings.data_dir / "bronze"
SILVER_LOCAL_DIR = settings.data_dir / "silver"
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

hf_token = settings.hf_token
SILVER_HF_REPO = (getattr(settings, "silver_hf_repo", None)
                  or settings.bronze_hf_repo.replace("bronze", "silver"))

STEP = pd.Timedelta(minutes=30)
TS = "timestamp_utc"  

# Absolute sanity bounds — a coarse "this is definitely garbage" net (negatives, absurd
# magnitudes). They are deliberately generous: the solar ceilings used to be 15 GW, which
# GB's fleet outgrew (capacity is now ~22 GW), so real sunny-midday peaks above 15 GW were
# silently deleted, punching NaN holes in the target that surfaced as dips in the forecast
# 1/2/3/7 days later via the lag features. Real quality control for the solar series is
# capacity-relative — see clamp_to_capacity() — which never goes stale as the fleet grows.
RANGES = {
    "generation_mw": (0.0, 60000.0),
    "embedded_solar_mw": (0.0, 60000.0),
    "embedded_wind_mw": (0.0, 60000.0),
    "ssrd_uk": (0.0, 1200.0),
    "tcc_uk": (0.0, 1.0),
    "lcc_uk": (0.0, 1.0),
    "t2m_uk": (220.0, 320.0),
    "ws10_uk": (0.0, 60.0),
    "ocf_total_mw": (0.0, 5000.0),
    "ocf_mean_wh": (0.0, 5000.0),
}
# time
def to_utc(series: pd.Series) -> pd.Series:
    """Parse to tz-aware UTC datetimes (accepts ISO 'Z' strings or tz-aware)."""
    out = pd.to_datetime(series, utc=True, errors="coerce")
    return out


def floor_30(series: pd.Series) -> pd.Series:
    return series.dt.floor("30min")


def canonical_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Continuous 30-min UTC grid covering [start, end] (period-start convention)."""
    return pd.date_range(start=start, end=end, freq="30min", tz="UTC", name=TS)


# schema
def enforce_float32(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def clamp_to_nan(df: pd.DataFrame, col: str) -> int:
    """Set out-of-range values to NaN. Returns count clamped (bad data)."""
    if col not in df.columns or col not in RANGES:
        return 0
    lo, hi = RANGES[col]
    bad = (df[col] < lo) | (df[col] > hi)
    n = int(bad.sum())
    if n:
        df.loc[bad, col] = np.nan
    return n

def clamp_to_capacity(df: pd.DataFrame, col: str, capacity_col: str,
                      factor: float = 1.5) -> int:
    """NaN values that are negative or exceed installed capacity * factor.

    Capacity-relative, so it scales with the fleet and never needs raising again — unlike a
    fixed MW ceiling, which silently deleted real peaks once GB solar outgrew it.

    `factor` is deliberately loose (1.5x): reported capacity lags actual installs, so a
    genuine reading can sit above it, and the cost of deleting a real peak (a NaN hole that
    propagates into the lag features days later) is far worse than keeping a slightly odd
    one. 1.5 also matches the gold contract's `target_cf` bound of [0, 1.5] — anything above
    that fails there anyway. For reference, the highest capacity factor ever observed in
    this dataset is ~0.74, so this leaves ~2x headroom while still catching real garbage
    (negatives, unit errors, order-of-magnitude spikes). Rows with unknown capacity fall
    back to the coarse RANGES net. Returns the count clamped.
    """
    if col not in df.columns or capacity_col not in df.columns:
        return 0
    cap = df[capacity_col]
    bad = (df[col] < 0) | (cap.notna() & (cap > 0) & (df[col] > cap * factor))
    n = int(bad.sum())
    if n:
        df.loc[bad, col] = np.nan
    return n

# missing-data policy
def apply_missing_policy(
    df: pd.DataFrame, value_col: str, max_ffill: int = 1, long_gap_steps: int = 6
) -> pd.DataFrame:
    """
    gap <= max_ffill steps (30 min): forward-fill, flag 'ffill'
    gap >  max_ffill and <= long_gap_steps: leave NaN, flag 'gap'
    gap >  long_gap_steps: leave NaN, flag 'long_gap'
    present value: flag 'ok'
    """
    df = df.sort_values(TS).reset_index(drop=True)
    present = df[value_col].notna()
    flag = pd.Series("ok", index=df.index, dtype="object")
    flag[~present] = "gap"

    # consecutive missing-run lengths
    run_id = (present != present.shift()).cumsum()
    run_len = df.groupby(run_id)[value_col].transform("size")
    miss = ~present
    flag[miss & (run_len > long_gap_steps)] = "long_gap"

    # forward-fill short gaps 
    fillable = miss & (run_len <= max_ffill)
    df[value_col] = df[value_col].ffill(limit=max_ffill)
    flag[fillable] = "ffill"

    df["data_quality_flag"] = flag.astype("string")
    return df


# spatial weights
def load_weights() -> dict[str, float]:
    """PV-capacity spatial weights per region from configs/uk_weather_points.csv."""
    cfg = CONFIGS_DIR / "uk_weather_points.csv"
    df = pd.read_csv(cfg)
    if "weight" not in df.columns:
        logger.warning("no 'weight' column; using equal weights")
        df["weight"] = 1.0
    w = dict(zip(df["region"].astype(str), df["weight"].astype(float)))
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}  

# IO
def read_bronze(source: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read all parquet for a Bronze source into one DataFrame (local only)."""
    base = BRONZE_LOCAL_DIR / source
    files = sorted(base.rglob("*.parquet"))
    if not files:
        logger.warning(f"no bronze parquet under {base}")
        return pd.DataFrame()
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    return pd.concat(frames, ignore_index=True)


def write_silver(df: pd.DataFrame, table: str) -> int:
    """Write a Silver table partitioned by year=YYYY/month=MM of timestamp_utc."""
    if df.empty:
        logger.warning(f"{table}: nothing to write")
        return 0
    import shutil
    shutil.rmtree(SILVER_LOCAL_DIR / table, ignore_errors=True)
    df = df.sort_values(TS).reset_index(drop=True)
    yr = df[TS].dt.year
    mo = df[TS].dt.month
    written = 0
    for (y, m), part in df.groupby([yr, mo]):
        out_dir = SILVER_LOCAL_DIR / table / f"year={int(y)}" / f"month={int(m):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{table}_{int(y)}{int(m):02d}.parquet"
        part.to_parquet(out_path, index=False, compression="snappy")
        written += len(part)
    logger.success(f"{table}: wrote {written:,} rows -> {SILVER_LOCAL_DIR / table}")
    return written


def read_silver(table: str) -> pd.DataFrame:
    base = SILVER_LOCAL_DIR / table
    files = sorted(base.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

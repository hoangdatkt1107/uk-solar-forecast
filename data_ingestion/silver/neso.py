"""Silver: NESO embedded wind & solar operator forecast (baseline)"""
from __future__ import annotations

import glob

import pandas as pd
from loguru import logger

from .common import (
    TS, BRONZE_LOCAL_DIR, to_utc, floor_30, canonical_index, enforce_float32,
    clamp_to_nan, apply_missing_policy, write_silver,
)
from .contracts import validate_neso

_REN = {
    "EMBEDDED_SOLAR_FORECAST": "embedded_solar_mw",
    "EMBEDDED_WIND_FORECAST": "embedded_wind_mw",
    "EMBEDDED_SOLAR_CAPACITY": "embedded_solar_capacity_mw",
    "EMBEDDED_WIND_CAPACITY": "embedded_wind_capacity_mw",
}
_VALCOLS = list(_REN.values())

def _load_neso_frames() -> pd.DataFrame:
    files = sorted(glob.glob(str(BRONZE_LOCAL_DIR / "neso" / "**" / "*.parquet"),
                             recursive=True))

    archives = [f for f in files if "archive" in f.lower()]
    if archives:
        files = archives
    else:
        logger.warning("neso: no archive files found, using whatever is present")
    frames = []
    for f in files:
        cols = ["DATE_GMT", "TIME_GMT"] + list(_REN.keys())
        df = pd.read_parquet(f)
        keep = [c for c in cols if c in df.columns]
        sub = df[keep].copy()
        sub["Forecast_Datetime"] = df["Forecast_Datetime"] if "Forecast_Datetime" in df.columns else pd.NaT
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def build_silver_neso() -> pd.DataFrame:
    raw = _load_neso_frames()
    if raw.empty:
        logger.warning("neso: no bronze data")
        return raw

    # timestamp from GMT date + time
    date = to_utc(raw["DATE_GMT"]).dt.normalize()
    tod = pd.to_timedelta(raw["TIME_GMT"].astype(str))
    ts = floor_30(date + tod)
    raw = raw.assign(**{TS: ts})
    raw["Forecast_Datetime"] = to_utc(raw["Forecast_Datetime"])

    # anti-leakage: keep latest issue <= slot 
    leak = raw["Forecast_Datetime"].notna() & (raw["Forecast_Datetime"] > raw[TS])
    raw = raw[~leak]
    raw = (raw.sort_values([TS, "Forecast_Datetime"], na_position="first")
              .drop_duplicates(TS, keep="last"))

    df = raw.rename(columns=_REN)
    df = enforce_float32(df, _VALCOLS)
    for c in ("embedded_solar_mw", "embedded_wind_mw"):
        clamp_to_nan(df, c)

    df = df.set_index(TS)[_VALCOLS]
    grid = canonical_index(df.index.min(), df.index.max())
    df = df.reindex(grid).reset_index()

    df = apply_missing_policy(df, "embedded_solar_mw", max_ffill=1)
    for c in _VALCOLS:
        df[c] = df[c].astype("float32")
    return df

def run() -> None:
    df = build_silver_neso()
    if df.empty:
        return
    validate_neso(df)
    write_silver(df, "silver_neso")

if __name__ == "__main__":
    run()

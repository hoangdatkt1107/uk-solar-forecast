"""Silver: NESO embedded wind & solar operator forecast (baseline)"""
from __future__ import annotations

import glob
import os
import re

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

def _neso_files() -> list[str]:
    """Archive parquet paths to build silver from (most-recent year only in LEAN mode)."""
    files = sorted(glob.glob(str(BRONZE_LOCAL_DIR / "neso" / "**" / "*.parquet"),
                             recursive=True))
    archives = [f for f in files if "archive" in f.lower()]
    if archives:
        files = archives
    else:
        logger.warning("neso: no archive files found, using whatever is present")

    # LEAN (serve): the NESO archive is ~38M rows across all years. Serve only needs recent
    # NESO (it's a same-slot feature, never lagged), so keep only the most-recent archive
    # year. Weekly retrain leaves this unset and builds the full history (still bounded —
    # see _reduce_neso_file).
    if os.getenv("GRIDSIGHT_NESO_LEAN", "0").strip() in ("1", "true", "True"):
        def _year(path: str) -> int:
            ys = re.findall(r"archive_(20\d{2})", path)
            return max((int(y) for y in ys), default=-1)
        max_year = max((_year(f) for f in files), default=-1)
        if max_year > 0:
            kept = [f for f in files if _year(f) == max_year]
            logger.info(f"neso LEAN: keeping {len(kept)}/{len(files)} archive file(s) "
                        f"for year {max_year}")
            files = kept
    return files

def _reduce_neso_file(f: str) -> pd.DataFrame:
    """Read ONE archive file and collapse it to one row per 30-min slot (latest issue at or
    before the slot). Doing the timestamp build + anti-leak + dedup per file means the raw
    ~5M-row-per-year archive never all lands in memory at once — each file reduces to ~17k
    rows before anything is concatenated, so even the full multi-year rebuild stays small."""
    cols = ["DATE_GMT", "TIME_GMT"] + list(_REN.keys())
    src = pd.read_parquet(f)
    keep = [c for c in cols if c in src.columns]
    raw = src[keep].copy()
    raw["Forecast_Datetime"] = src["Forecast_Datetime"] if "Forecast_Datetime" in src.columns else pd.NaT
    del src

    # timestamp from GMT date + time
    date = to_utc(raw["DATE_GMT"]).dt.normalize()
    tod = pd.to_timedelta(raw["TIME_GMT"].astype(str))
    raw[TS] = floor_30(date + tod)
    raw["Forecast_Datetime"] = to_utc(raw["Forecast_Datetime"])

    # anti-leakage: keep latest issue <= slot
    leak = raw["Forecast_Datetime"].notna() & (raw["Forecast_Datetime"] > raw[TS])
    raw = raw[~leak]
    raw = (raw.sort_values([TS, "Forecast_Datetime"], na_position="first")
              .drop_duplicates(TS, keep="last"))
    return raw.rename(columns=_REN).reindex(columns=[TS] + _VALCOLS)

def build_silver_neso() -> pd.DataFrame:
    files = _neso_files()
    reduced = [_reduce_neso_file(f) for f in files]
    reduced = [r for r in reduced if not r.empty]
    if not reduced:
        logger.warning("neso: no bronze data")
        return pd.DataFrame()

    # year archives don't share slots, but coalesce any boundary dupes just in case
    df = pd.concat(reduced, ignore_index=True).drop_duplicates(TS, keep="last")
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

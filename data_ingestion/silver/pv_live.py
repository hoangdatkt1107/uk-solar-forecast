"""Silver: PV_Live national generation (the TARGET)"""
from __future__ import annotations
import pandas as pd
from loguru import logger
from .common import (
    TS, STEP, to_utc, floor_30, canonical_index, enforce_float32, clamp_to_nan,
    apply_missing_policy, read_bronze, write_silver,
)
from .contracts import validate_pv_live

def build_silver_pv_live() -> pd.DataFrame:
    raw = read_bronze("pv_live", columns=["gsp_id", "datetime_gmt", "generation_mw",
                                          "capacity_mwp", "fetched_at"])
    if raw.empty:
        logger.warning("pv_live: no bronze data")
        return raw

    df = raw[raw["gsp_id"] == 0].copy() 
    # period-END -> period-START, normalise to 30-min UTC grid
    ts = floor_30(to_utc(df["datetime_gmt"]) - STEP)
    df = df.assign(**{TS: ts})

    df = enforce_float32(df, ["generation_mw", "capacity_mwp"])
    n_neg = clamp_to_nan(df, "generation_mw")  
    if n_neg:
        logger.info(f"pv_live: {n_neg} out-of-range generation values -> NaN")

    # one row per slot: prefer the freshest fetch
    df = (df.sort_values([TS, "fetched_at"])
            .drop_duplicates(TS, keep="last")
            .set_index(TS))

    grid = canonical_index(df.index.min(), df.index.max())
    df = df.reindex(grid)[["generation_mw", "capacity_mwp"]].reset_index()
    df["capacity_mwp"] = df["capacity_mwp"].ffill().bfill()  

    df = apply_missing_policy(df, "generation_mw", max_ffill=1)
    df["generation_mw"] = df["generation_mw"].astype("float32")
    df["capacity_mwp"] = df["capacity_mwp"].astype("float32")
    return df

def run() -> None:
    df = build_silver_pv_live()
    if df.empty:
        return
    validate_pv_live(df)
    write_silver(df, "silver_pv_live")

if __name__ == "__main__":
    run()

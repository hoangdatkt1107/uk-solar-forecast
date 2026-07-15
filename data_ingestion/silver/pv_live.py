"""Silver: PV_Live national generation (the TARGET)"""
from __future__ import annotations
import pandas as pd
from loguru import logger
from .common import (
    TS, STEP, to_utc, floor_30, canonical_index, enforce_float32, clamp_to_nan,
    clamp_to_capacity, apply_missing_policy, read_bronze, write_silver,
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
    # Quality control against installed capacity, not a fixed MW ceiling: generation can't
    # meaningfully exceed the fleet, and this keeps working as the fleet grows. (A hardcoded
    # 15 GW ceiling used to delete real >15 GW summer peaks — those NaNs then propagated
    # into the lag features and dented the forecast 1/2/3/7 days later.)
    n_bad = clamp_to_nan(df, "generation_mw")            # coarse net (negatives / absurd)
    n_bad += clamp_to_capacity(df, "generation_mw", "capacity_mwp")
    if n_bad:
        logger.info(f"pv_live: {n_bad} out-of-range generation values -> NaN")

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

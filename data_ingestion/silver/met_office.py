"""Silver: Met Office NWP -> single UK-weighted weather series (silver_met_office_nwp).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .common import (
    TS, to_utc, canonical_index, apply_missing_policy, clamp_to_nan,
    load_weights, read_bronze, write_silver,
)
from .contracts import validate_met_office_nwp

_VARS = ["ssrd", "tcc", "lcc", "t2m", "ws10"]


def build_silver_met_office_nwp() -> pd.DataFrame:
    df = read_bronze("met_office_nwp",
                     columns=["init_time", "valid_time", "region"] + _VARS)
    if df.empty:
        logger.warning("met_office_nwp: no bronze data")
        return df

    df["init_time"] = to_utc(df["init_time"])
    df["valid_time"] = to_utc(df["valid_time"])
    df = df[df["init_time"] <= df["valid_time"]]  # anti-leakage guard

    # latest available init_time per valid_time
    latest = df.groupby("valid_time")["init_time"].transform("max")
    df = df[df["init_time"] == latest]

    # PV-capacity weighted spatial average across regions
    weights = load_weights()
    df["w"] = df["region"].map(weights).fillna(0.0)
    g = df.groupby("valid_time")
    out = pd.DataFrame(index=g.size().index)
    for v in _VARS:
        wsum = g.apply(lambda x, v=v: np.average(x[v], weights=x["w"])
                       if x["w"].sum() > 0 else np.nan)
        out[f"{v}_uk"] = wsum
    out["init_time"] = g["init_time"].first()
    out = out.reset_index().rename(columns={"valid_time": TS})

    # forecast age
    out["nwp_age_h"] = ((out[TS] - out["init_time"]).dt.total_seconds() / 3600.0)

    # hourly -> 30-min grid, ffill within the hour (cap 1 step)
    out = out.set_index(TS)
    grid = canonical_index(out.index.min(), out.index.max())
    out = out.reindex(grid)
    fill_cols = [f"{v}_uk" for v in _VARS] + ["init_time", "nwp_age_h"]
    out[fill_cols] = out[fill_cols].ffill(limit=1)
    out = out.reset_index()

    for v in _VARS:
        clamp_to_nan(out, f"{v}_uk")
        out[f"{v}_uk"] = out[f"{v}_uk"].astype("float32")
    out["nwp_age_h"] = out["nwp_age_h"].astype("float32")

    out = apply_missing_policy(out, "ssrd_uk", max_ffill=1)
    return out

def run() -> None:
    df = build_silver_met_office_nwp()
    if df.empty:
        return
    validate_met_office_nwp(df)
    write_silver(df, "silver_met_office_nwp")

if __name__ == "__main__":
    run()

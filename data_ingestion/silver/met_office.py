"""Silver: Met Office NWP -> single UK-weighted weather series (silver_met_office_nwp).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .common import (
    TS, to_utc, clamp_to_nan, load_weights, read_bronze, write_silver,
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

    # Keep the full (valid_time x init_time) grid: Gold picks the as-of init per horizon.
    # PV-capacity weighted spatial average across regions, per (valid, init).
    weights = load_weights()
    df["w"] = df["region"].map(weights).fillna(0.0)
    g = df.groupby(["valid_time", "init_time"])
    out = pd.DataFrame(index=g.size().index)
    for v in _VARS:
        out[f"{v}_uk"] = g.apply(lambda x, v=v: np.average(x[v], weights=x["w"])
                                 if x["w"].sum() > 0 else np.nan)
    out = out.reset_index().rename(columns={"valid_time": TS})

    out["nwp_age_h"] = ((out[TS] - out["init_time"]).dt.total_seconds() / 3600.0).astype("float32")
    for v in _VARS:
        clamp_to_nan(out, f"{v}_uk")
        out[f"{v}_uk"] = out[f"{v}_uk"].astype("float32")
    return out

def run() -> None:
    df = build_silver_met_office_nwp()
    if df.empty:
        return
    validate_met_office_nwp(df)
    write_silver(df, "silver_met_office_nwp")

if __name__ == "__main__":
    run()

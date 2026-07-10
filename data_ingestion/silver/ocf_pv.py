"""Silver: OCF UK PV (per-system rooftop sample) -> fleet solar index"""
from __future__ import annotations
import glob
import pandas as pd
from loguru import logger

from .common import (
    TS, BRONZE_LOCAL_DIR, to_utc, floor_30, canonical_index,
    apply_missing_policy, clamp_to_nan, write_silver,
)
from .contracts import validate_ocf_pv

def build_silver_ocf_pv() -> pd.DataFrame:
    files = sorted(glob.glob(str(BRONZE_LOCAL_DIR / "ocf_pv" / "**" / "*.parquet"),
                             recursive=True))
    if not files:
        logger.warning("ocf_pv: no bronze data")
        return pd.DataFrame()

    # streaming aggregate
    acc_sum = pd.Series(dtype="float64")
    acc_cnt = pd.Series(dtype="float64")
    for i, f in enumerate(files, 1):
        d = pd.read_parquet(f, columns=["datetime_GMT", "generation_Wh"])
        ts = floor_30(to_utc(d["datetime_GMT"]))
        g = d.assign(**{TS: ts}).groupby(TS)["generation_Wh"].agg(["sum", "count"])
        acc_sum = acc_sum.add(g["sum"], fill_value=0.0)
        acc_cnt = acc_cnt.add(g["count"], fill_value=0.0)
        if i % 50 == 0 or i == len(files):
            logger.info(f"ocf_pv: aggregated {i}/{len(files)} files")

    df = pd.DataFrame({"total_wh": acc_sum, "n": acc_cnt}).sort_index()
    df.index.name = TS
    df = df.reset_index()
    df["ocf_total_mw"] = (df["total_wh"] / 5e5).astype("float32")   # Wh/0.5h -> MW
    df["ocf_mean_wh"] = (df["total_wh"] / df["n"].clip(lower=1)).astype("float32")
    df["ocf_n_systems"] = df["n"].astype("int32")
    df = df[[TS, "ocf_total_mw", "ocf_mean_wh", "ocf_n_systems"]]

    # canonical grid + quality flags
    df = df.set_index(TS)
    grid = canonical_index(df.index.min(), df.index.max())
    df = df.reindex(grid).reset_index()
    clamp_to_nan(df, "ocf_total_mw")
    df = apply_missing_policy(df, "ocf_total_mw", max_ffill=1)
    df["ocf_total_mw"] = df["ocf_total_mw"].astype("float32")
    df["ocf_mean_wh"] = df["ocf_mean_wh"].astype("float32")
    df["ocf_n_systems"] = df["ocf_n_systems"].astype("Int32")
    return df

def run() -> None:
    df = build_silver_ocf_pv()
    if df.empty:
        return
    validate_ocf_pv(df)
    write_silver(df, "silver_ocf_pv")

if __name__ == "__main__":
    run()

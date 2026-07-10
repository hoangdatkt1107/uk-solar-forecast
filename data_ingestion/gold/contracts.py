"""Gold data-quality + anti-leakage contracts (hard-fail on violation)"""
from __future__ import annotations
import re
import pandas as pd
from loguru import logger

from .common import TS
from .lag_features import lag_set


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def validate_gold(df: pd.DataFrame, horizon: int) -> None:
    # time grid
    _assert(df[TS].notna().all(), f"{TS} has nulls")
    _assert(str(df[TS].dt.tz) == "UTC", f"{TS} not tz-aware UTC")
    _assert(df[TS].is_unique, f"{TS} not unique")
    _assert(df[TS].is_monotonic_increasing, f"{TS} not sorted")
    steps = df[TS].diff().dropna().value_counts()
    _assert(steps.index[0] == pd.Timedelta(minutes=30), "spine not on 30-min grid")

    # targets
    for c in ("target_mw", "target_cf"):
        _assert(c in df.columns, f"missing target {c}")
    cf = df["target_cf"].dropna()
    _assert(((cf >= 0) & (cf <= 1.5)).all(), "target_cf out of [0,1.5]")
    _assert((df["target_mw"].dropna() >= 0).all(), "negative target_mw")

    # ANTI-LEAKAGE: raw observed-at-t actuals must NOT be present as features
    leaky = {"generation_mw", "ocf_total_mw", "ocf_mean_wh", "ocf_n_systems"}
    present = leaky & set(df.columns)
    _assert(not present, f"LEAKAGE: raw observed columns present as features: {present}")

    # every observed-actual lag column must be shifted >= horizon
    lag_cols = [c for c in df.columns if re.search(r"_lag_(\d+)$", c)]
    for c in lag_cols:
        L = int(re.search(r"_lag_(\d+)$", c).group(1))
        _assert(L >= horizon, f"LEAKAGE: {c} uses lag {L} < horizon {horizon}")
    _assert(set(lag_set(horizon)).issubset(
        {int(re.search(r'gen_lag_(\d+)', c).group(1))
         for c in df.columns if c.startswith("gen_lag_")}),
        "missing expected gen lag columns")

    if "solar_elevation_deg" in df:
        e = df["solar_elevation_deg"]
        _assert(((e >= -90) & (e <= 90)).all(), "solar_elevation out of range")
        # PV should be ~0 when sun is well below horizon
        night = df.loc[e < -5, "target_mw"].dropna()
        if len(night):
            frac0 = (night < 50).mean()
            (logger.warning if frac0 < 0.99 else logger.info)(
                f"night (elev<-5): {frac0:.1%} of slots have gen<50MW")

    logger.success(f"contract OK: gold_features (horizon={horizon}, {df.shape[1]} cols)")

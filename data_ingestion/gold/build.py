"""Orchestrate the Gold feature store: merge -> targets -> calendar/solar -> lags"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from .common import TS, write_gold, GOLD_TABLE
from .merge import merge_silver
from .targets import add_targets
from .calendar_features import add_calendar, add_solar
from .lag_features import add_lags_rolling
from .contracts import validate_gold

DEFAULT_HORIZON = 48  # 30-min steps = 24h (day-ahead)

# column order: keys, targets, then feature groups
_LEAD = [TS, "target_mw", "target_cf", "capacity_mwp"]

_LEAKY_OBSERVED = ["generation_mw", "ocf_total_mw", "ocf_mean_wh", "ocf_n_systems"]


def build_gold(horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    df = merge_silver()
    if df.empty:
        return df
    logger.info(f"gold: building features (horizon={horizon} steps = {horizon/2:.0f}h)")
    df = add_targets(df)         
    df = add_calendar(df)
    df = add_solar(df)
    df = add_lags_rolling(df, horizon)   

    # drop raw observed actuals now that their leakage-safe lags exist
    df = df.drop(columns=[c for c in _LEAKY_OBSERVED if c in df.columns])

    # tidy column order
    front = [c for c in _LEAD if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    df.attrs["horizon"] = horizon
    return df


def run(horizon: int = DEFAULT_HORIZON, upload: bool = False) -> None:
    df = build_gold(horizon)
    if df.empty:
        return
    validate_gold(df, horizon)
    write_gold(df, GOLD_TABLE)
    if upload:
        from .upload import upload_gold_to_hf
        upload_gold_to_hf()


if __name__ == "__main__":
    run()

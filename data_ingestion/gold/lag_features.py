from __future__ import annotations
import pandas as pd
from .common import TS

# candidate lags (in 30-min steps): same time yesterday / 2d / 3d / 1 week
_DAILY = [48, 96, 144, 336]


def lag_set(horizon: int) -> list[int]:
    """Lags >= horizon: the horizon itself plus the standard daily/weekly ones."""
    return sorted({horizon} | {l for l in _DAILY if l >= horizon})


def add_lags_rolling(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Requires a continuous 30-min spine (pv_live is gap-free) so shift == time shift."""
    df = df.sort_values(TS).reset_index(drop=True).copy()
    gen, cf, ocf = df["generation_mw"], df["target_cf"], df.get("ocf_total_mw")

    for L in lag_set(horizon):
        df[f"gen_lag_{L}"] = gen.shift(L).astype("float32")
        df[f"cf_lag_{L}"] = cf.shift(L).astype("float32")

    gh, cfh = gen.shift(horizon), cf.shift(horizon)
    df["gen_roll_mean_48"] = gh.rolling(48, min_periods=24).mean().astype("float32")
    df["gen_roll_mean_336"] = gh.rolling(336, min_periods=168).mean().astype("float32")
    df["gen_roll_std_48"] = gh.rolling(48, min_periods=24).std().astype("float32")
    df["cf_roll_mean_48"] = cfh.rolling(48, min_periods=24).mean().astype("float32")
    df["cf_roll_mean_336"] = cfh.rolling(336, min_periods=168).mean().astype("float32")

    if ocf is not None:
        oh = ocf.shift(horizon)
        df[f"ocf_lag_{horizon}"] = oh.astype("float32")
        df["ocf_roll_mean_48"] = oh.rolling(48, min_periods=24).mean().astype("float32")

    warmup = max(lag_set(horizon)) + 1
    df["has_full_history"] = (df.index >= warmup).astype("int8")
    return df

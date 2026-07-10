from __future__ import annotations
import numpy as np
import pandas as pd


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_mw"] = df["generation_mw"].astype("float32")
    cap = df["capacity_mwp"].replace(0, np.nan)
    df["target_cf"] = (df["generation_mw"] / cap).clip(0, 1.5).astype("float32")
    return df

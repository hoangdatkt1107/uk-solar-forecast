from __future__ import annotations
import pandas as pd
from loguru import logger
from .common import TS, read_silver

def _norm_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Force timestamp_utc to a single resolution/tz so joins match exactly"""
    df = df.copy()
    df[TS] = pd.to_datetime(df[TS], utc=True).astype("datetime64[us, UTC]")
    return df

def _nwp_asof(nwp: pd.DataFrame, horizon: int) -> pd.DataFrame:
    nwp = nwp.copy()
    nwp["init_time"] = pd.to_datetime(nwp["init_time"], utc=True)
    nwp = nwp[nwp["init_time"] <= nwp[TS] - pd.Timedelta(minutes=30 * horizon)]
    keep = nwp.groupby(TS)["init_time"].idxmax()
    return nwp.loc[keep].drop(columns="init_time")

def merge_silver(horizon: int, extend_to: pd.Timestamp | None = None) -> pd.DataFrame:
    pv = read_silver("silver_pv_live")
    nwp = read_silver("silver_met_office_nwp")
    neso = read_silver("silver_neso")
    if pv.empty:
        logger.error("merge: silver_pv_live is empty - build Silver first")
        return pd.DataFrame()

    pv = (_norm_ts(pv)
          .rename(columns={"data_quality_flag": "pv_flag"})
          .drop_duplicates(TS).sort_values(TS).reset_index(drop=True))

    if extend_to is not None:                        # future rows for live serving
        fut = pd.date_range(pv[TS].max() + pd.Timedelta(minutes=30),
                            extend_to, freq="30min")
        if len(fut):
            add = pd.DataFrame({TS: fut})
            add["capacity_mwp"] = pv["capacity_mwp"].iloc[-1]   # carry latest capacity
            pv = _norm_ts(pd.concat([pv, add], ignore_index=True))

    nwp = _nwp_asof(_norm_ts(nwp), horizon).drop_duplicates(TS)
    neso = (_norm_ts(neso)
            .rename(columns={"data_quality_flag": "neso_flag"})
            .drop_duplicates(TS))

    out = pv
    for src in (nwp, neso):
        out = out.merge(src, on=TS, how="left")

    out = out.sort_values(TS).reset_index(drop=True)
    nwp_cols = [c for c in out.columns if c.endswith("_uk")] + ["nwp_age_h"]
    out[nwp_cols] = out[nwp_cols].ffill(limit=1)   # hourly NWP -> 30-min spine

    for col, name in [("ssrd_uk", "nwp"), ("embedded_solar_mw", "neso")]:
        if col in out:
            logger.info(f"merge: {name} coverage = {out[col].notna().mean()*100:.1f}% "
                        f"of {len(out):,} spine slots")
    return out

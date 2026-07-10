from __future__ import annotations
import pandas as pd
from loguru import logger
from .common import TS, read_silver


def _norm_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Force timestamp_utc to a single resolution/tz so joins match exactly."""
    df = df.copy()
    df[TS] = pd.to_datetime(df[TS], utc=True).astype("datetime64[us, UTC]")
    return df


def merge_silver() -> pd.DataFrame:
    pv = read_silver("silver_pv_live")
    nwp = read_silver("silver_met_office_nwp")
    neso = read_silver("silver_neso")
    ocf = read_silver("silver_ocf_pv")
    if pv.empty:
        logger.error("merge: silver_pv_live is empty - build Silver first")
        return pd.DataFrame()

    pv = (_norm_ts(pv)
          .rename(columns={"data_quality_flag": "pv_flag"})
          .drop_duplicates(TS).sort_values(TS).reset_index(drop=True))

    nwp = (_norm_ts(nwp)
           .drop(columns=["init_time"], errors="ignore")
           .rename(columns={"data_quality_flag": "nwp_flag"})
           .drop_duplicates(TS))
    neso = (_norm_ts(neso)
            .rename(columns={"data_quality_flag": "neso_flag"})
            .drop_duplicates(TS))
    ocf = (_norm_ts(ocf)
           .rename(columns={"data_quality_flag": "ocf_flag"})
           .drop_duplicates(TS))

    out = pv
    for src in (nwp, neso, ocf):
        out = out.merge(src, on=TS, how="left")

    out = out.sort_values(TS).reset_index(drop=True)

    for col, name in [("ssrd_uk", "nwp"), ("embedded_solar_mw", "neso"),
                      ("ocf_total_mw", "ocf")]:
        if col in out:
            logger.info(f"merge: {name} coverage = {out[col].notna().mean()*100:.1f}% "
                        f"of {len(out):,} spine slots")
    return out

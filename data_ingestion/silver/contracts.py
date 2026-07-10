from __future__ import annotations
import pandas as pd
from loguru import logger
from .common import TS, RANGES, read_silver


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)

def _check_utc_index(df: pd.DataFrame) -> None:
    _assert(TS in df.columns, f"missing {TS}")
    _assert(df[TS].notna().all(), f"{TS} has nulls")
    _assert(str(df[TS].dt.tz) == "UTC", f"{TS} not tz-aware UTC (got {df[TS].dt.tz})")
    _assert(df[TS].is_unique, f"{TS} not unique (duplicate slots)")


def _check_range(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns or col not in RANGES:
        return
    lo, hi = RANGES[col]
    v = df[col].dropna()
    _assert(((v >= lo) & (v <= hi)).all(),
            f"{col} out of range [{lo},{hi}] (min={v.min()}, max={v.max()})")

def _coverage(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    full = pd.date_range(df[TS].min(), df[TS].max(), freq="30min", tz="UTC")
    return len(df) / len(full)


def validate_pv_live(df: pd.DataFrame) -> None:
    _check_utc_index(df)
    _check_range(df, "generation_mw")
    _assert((df["generation_mw"].dropna() >= 0).all(), "generation_mw has negatives")
    cap = df["capacity_mwp"]
    valid = df["generation_mw"].notna() & cap.notna()
    if valid.any():
        ok = (df.loc[valid, "generation_mw"] <= cap[valid] * 1.05).mean()
        if ok < 0.95:
            logger.warning(f"pv_live: only {ok:.1%} rows <= 1.05*capacity (<95%)")
    cov = _coverage(df)
    (logger.warning if cov < 0.98 else logger.info)(f"pv_live coverage={cov:.1%}")
    logger.success("contract OK: silver_pv_live")


def validate_met_office_nwp(df: pd.DataFrame) -> None:
    _check_utc_index(df)
    for c in ("ssrd_uk", "tcc_uk", "lcc_uk", "t2m_uk", "ws10_uk"):
        _check_range(df, c)
    if "init_time" in df.columns:
        m = df["init_time"].notna()
        _assert((df.loc[m, "init_time"] <= df.loc[m, TS]).all(),
                "init_time > valid_time (leakage!)")
    if "nwp_age_h" in df.columns:
        _assert((df["nwp_age_h"].dropna() >= 0).all(), "negative nwp_age_h")
    logger.success("contract OK: silver_met_office_nwp")


def validate_ocf_pv(df: pd.DataFrame) -> None:
    _check_utc_index(df)
    _check_range(df, "ocf_total_mw")
    _assert((df["ocf_total_mw"].dropna() >= 0).all(), "ocf_total_mw negative")
    if "ocf_n_systems" in df.columns:
        _assert((df["ocf_n_systems"].dropna() >= 0).all(), "ocf_n_systems negative")
    logger.success("contract OK: silver_ocf_pv")


def validate_neso(df: pd.DataFrame) -> None:
    _check_utc_index(df)
    _check_range(df, "embedded_solar_mw")
    _assert((df["embedded_solar_mw"].dropna() >= 0).all(), "embedded_solar_mw negative")
    logger.success("contract OK: silver_neso")

def cross_source_checks() -> None:
    """Soft sanity checks across tables (warnings only)."""
    pv = read_silver("silver_pv_live")
    neso = read_silver("silver_neso")
    if not pv.empty and not neso.empty:
        m = pv[[TS, "generation_mw"]].merge(
            neso[[TS, "embedded_solar_mw"]], on=TS, how="inner"
        ).dropna()
        if len(m) > 100:
            corr = m["generation_mw"].corr(m["embedded_solar_mw"])
            (logger.warning if corr < 0.85 else logger.success)(
                f"corr(pv_live.generation, neso.embedded_solar) = {corr:.3f} "
                f"over {len(m):,} slots (expect > 0.85)"
            )
        else:
            logger.info(f"cross-check pv/neso: only {len(m)} overlapping slots, skipping")

"""Clear-sky GHI — a physical prior fed straight into the meta-learner.

Gold already provides solar geometry (`solar_elevation_deg`, `clearsky_cos`).
Here we turn elevation into an approximate clear-sky global horizontal
irradiance (W/m^2) with the Haurwitz model — a simple, dependency-free clear-sky
model that depends only on the solar zenith angle:

    GHI_clear = 1098 * cos(zenith) * exp(-0.059 / cos(zenith)),   cos(zenith) > 0

This gives the meta-learner a calibrated "ceiling" for how much sun is physically
possible at each instant, which helps re-scale the base quantiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clearsky_ghi_from_elevation(elevation_deg: np.ndarray) -> np.ndarray:
    elev = np.asarray(elevation_deg, dtype="float64")
    cosz = np.clip(np.sin(np.radians(elev)), 0.0, 1.0)   # cos(zenith) = sin(elevation)
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi = 1098.0 * cosz * np.exp(-0.059 / np.where(cosz > 0, cosz, np.nan))
    return np.nan_to_num(ghi, nan=0.0).astype("float32")


def clearsky_feature(df: pd.DataFrame) -> np.ndarray:
    """Return clear-sky GHI for each row, using Gold's solar_elevation_deg if present."""
    if "solar_elevation_deg" in df.columns:
        return clearsky_ghi_from_elevation(df["solar_elevation_deg"].to_numpy())
    if "clearsky_cos" in df.columns:                      # fallback: scale cos(zenith)
        return (1098.0 * df["clearsky_cos"].to_numpy()).astype("float32")
    raise KeyError("Gold has neither solar_elevation_deg nor clearsky_cos")

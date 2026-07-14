from __future__ import annotations
import numpy as np
import pandas as pd
from .common import TS, UK_LAT, UK_LON

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[TS].dt
    df["hour"] = ts.hour.astype("int16")
    df["half_hour"] = (ts.hour * 2 + (ts.minute // 30)).astype("int16")  
    df["dow"] = ts.dayofweek.astype("int16")
    df["month"] = ts.month.astype("int16")
    df["doy"] = ts.dayofyear.astype("int16")
    df["is_weekend"] = (ts.dayofweek >= 5).astype("int8")

    hh = df["half_hour"].to_numpy()
    doy = df["doy"].to_numpy()
    df["tod_sin"] = np.sin(2 * np.pi * hh / 48).astype("float32")
    df["tod_cos"] = np.cos(2 * np.pi * hh / 48).astype("float32")
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    return df

def solar_position(index: pd.DatetimeIndex, lat: float, lon: float):
    """NOAA solar position -> (elevation_deg, cos_zenith>=0). Vectorised."""
    idx = pd.DatetimeIndex(index)
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    # equation of time (minutes) and solar declination (radians)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    time_offset = eqtime + 4.0 * lon            
    tst = hour * 60.0 + time_offset             
    ha = np.radians(tst / 4.0 - 180.0)         
    lat_r = np.radians(lat)
    cos_zen = (np.sin(lat_r) * np.sin(decl)
               + np.cos(lat_r) * np.cos(decl) * np.cos(ha))
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    elevation = np.degrees(np.arcsin(cos_zen))
    return elevation, np.clip(cos_zen, 0.0, None)


def add_solar(df: pd.DataFrame, lat: float = UK_LAT, lon: float = UK_LON) -> pd.DataFrame:
    df = df.copy()
    elev, csky = solar_position(df[TS], lat, lon)
    df["solar_elevation_deg"] = elev.astype("float32")
    df["clearsky_cos"] = csky.astype("float32")     
    df["is_daylight"] = (elev > 0).astype("int8")
    return df

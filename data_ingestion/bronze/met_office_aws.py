"""Met Office UK 2km deterministic (AWS Open Data) -> Bronze parquet (point-extracted)

Same schema/units as the HF met_office bronze, so Silver reads it unchanged.
One parquet per init-time; reruns skip finished inits (resumable).
"""
from __future__ import annotations
import time
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .common import BRONZE_LOCAL_DIR, CONFIGS_DIR, _ensure_partition_dir

S3_BASE = "https://met-office-atmospheric-model-data.s3.eu-west-2.amazonaws.com"
MODEL = "uk-deterministic-2km"

# our column -> Met Office variable file suffix on S3
_VARMAP = {
    "ssrd": "radiation_flux_in_shortwave_total_downward_at_surface",
    "tcc":  "cloud_amount_of_total_cloud",
    "lcc":  "cloud_amount_of_low_cloud",
    "t2m":  "temperature_at_screen_level",
    "ws10": "wind_speed_at_10m",
}

_CRS_CF = {
    "grid_mapping_name": "lambert_azimuthal_equal_area",
    "latitude_of_projection_origin": 54.9,
    "longitude_of_projection_origin": -2.5,
    "false_easting": 0.0,
    "false_northing": 0.0,
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.314140356,
}

_DEFAULT_POINTS = [
    ("South_East", 51.20, 0.50), ("London", 51.51, -0.13),
    ("East_Anglia", 52.40, 0.90), ("South_West", 50.80, -3.50),
    ("Midlands", 52.48, -1.90), ("North", 54.00, -1.50),
    ("Scotland", 56.50, -4.20),
]


def _load_uk_points() -> list[tuple[str, float, float]]:
    cfg = CONFIGS_DIR / "uk_weather_points.csv"
    if not cfg.exists():
        logger.warning(f"{cfg} not found; using built-in default UK points")
        return _DEFAULT_POINTS
    df = pd.read_csv(cfg)
    return [(str(r.region), float(r.lat), float(r.lon)) for r in df.itertuples()]


def _grid_index(ds, points: list[tuple[str, float, float]]) -> list[tuple[int, int]]:
    from pyproj import CRS, Transformer
    tf = Transformer.from_crs("EPSG:4326", CRS.from_cf(_CRS_CF), always_xy=True)
    xc = np.asarray(ds["projection_x_coordinate"].values)
    yc = np.asarray(ds["projection_y_coordinate"].values)
    idx = []
    for _region, lat, lon in points:
        gx, gy = tf.transform(lon, lat)
        idx.append((int(np.abs(yc - gy).argmin()), int(np.abs(xc - gx).argmin())))
    return idx


def _init_times(start: str, end: str, step_h: int) -> list[pd.Timestamp]:
    s = pd.Timestamp(start, tz="UTC").floor("h")
    s = s.replace(hour=(s.hour // step_h) * step_h)
    e = pd.Timestamp(end, tz="UTC")
    out, t = [], s
    while t <= e:
        out.append(t)
        t = t + pd.Timedelta(hours=step_h)
    return out


def _fetch(client, url: str, dst: Path) -> bool:
    """Stream one .nc to dst. False on 404 (missing lead/init). Retries transient errors."""
    for attempt in range(3):
        try:
            with client.stream("GET", url, timeout=120, follow_redirects=True) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(dst, "wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            return True
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return False


def _extract_init(init: pd.Timestamp, points, cols, max_lead_h: int, overwrite: bool) -> str:
    import httpx
    import xarray as xr

    out_path = (_ensure_partition_dir("met_office_nwp", init.year, init.month)
                / f"nwp_{init:%Y%m%dT%H%M}Z.parquet")
    if not overwrite and out_path.exists() and out_path.stat().st_size > 256:
        return "skip"

    tmp = Path(tempfile.mkdtemp(prefix="metaws_"))
    dst = tmp / "f.nc"
    idx = None
    rows = []
    try:
        with httpx.Client(headers={"User-Agent": "gridsight-bronze/1.0"}) as client:
            for lead in range(max_lead_h + 1):
                valid = init + pd.Timedelta(hours=lead)
                vals = {r: {} for r, _, _ in points}
                for col in cols:
                    url = (f"{S3_BASE}/{MODEL}/{init:%Y%m%dT%H%M}Z/"
                           f"{valid:%Y%m%dT%H%M}Z-PT{lead:04d}H00M-{_VARMAP[col]}.nc")
                    if not _fetch(client, url, dst):
                        continue
                    ds = xr.open_dataset(dst)
                    if idx is None:
                        idx = _grid_index(ds, points)
                    main = next(v for v in ds.data_vars if {"projection_y_coordinate",
                                "projection_x_coordinate"} <= set(ds[v].dims))
                    grid = np.asarray(ds[main].values)
                    ds.close()
                    for (region, _lat, _lon), (yi, xi) in zip(points, idx):
                        vals[region][col] = float(grid[yi, xi])
                    dst.unlink(missing_ok=True)

                now = datetime.now(timezone.utc).isoformat()
                for region, lat, lon in points:
                    if not vals[region]:
                        continue
                    rows.append({"init_time": init, "valid_time": valid, "forecast_hour": lead,
                                 "region": region, "lat": lat, "lon": lon,
                                 **{c: vals[region].get(c) for c in cols},
                                 "source": "aws_uk2km", "extracted_at": now})
        if not rows:
            return "empty"
        pd.DataFrame(rows).to_parquet(out_path, index=False, compression="snappy")
        return "done"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ingest_met_office_aws(
    start: str,
    end: str,
    init_step_h: int = 3,
    max_lead_h: int = 15,
    workers: int = 6,
    overwrite: bool = False,
    cols: list[str] | None = None,
) -> None:
    try:
        import httpx  
        import xarray  
        import netCDF4  
        import pyproj  
    except ImportError as e:
        logger.error(f"missing dep: {e}. pip install httpx xarray netCDF4 pyproj pyarrow")
        return
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cols = cols or list(_VARMAP)
    points = _load_uk_points()
    inits = _init_times(start, end, init_step_h)
    month_total = Counter(f"{i:%Y-%m}" for i in inits)
    month_seen: Counter = Counter()
    logger.info(f"Met Office AWS UK-2km | {inits[0]:%Y-%m-%d} -> {inits[-1]:%Y-%m-%d} "
                f"inits={len(inits)} months={len(month_total)} step={init_step_h}h "
                f"max_lead={max_lead_h}h vars={cols} points={len(points)} workers={workers}")

    done = skipped = empty = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_init, i, points, cols, max_lead_h, overwrite): i
                   for i in inits}
        for n, fut in enumerate(as_completed(futures), 1):
            init = futures[fut]
            try:
                st = fut.result()
                done += st == "done"
                skipped += st == "skip"
                empty += st == "empty"
            except Exception as e:
                failed += 1
                logger.error(f"  {init:%Y%m%dT%H%M}Z failed: {e}")
            key = f"{init:%Y-%m}"
            month_seen[key] += 1
            if month_seen[key] == month_total[key]:
                logger.info(f"month {key} finished ({month_total[key]} inits) | "
                            f"overall {n}/{len(inits)} done={done} skip={skipped} fail={failed}")
            elif n % 200 == 0 or n == len(inits):
                logger.info(f"progress {n}/{len(inits)} done={done} skip={skipped} "
                            f"empty={empty} fail={failed}")

    logger.success(f"Met Office AWS complete: done={done} skipped={skipped} empty={empty} "
                   f"failed={failed} -> {BRONZE_LOCAL_DIR / 'met_office_nwp'}")

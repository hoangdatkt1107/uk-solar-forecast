"""Met Office UK NWP (HuggingFace) -> Bronze
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from .common import BRONZE_LOCAL_DIR, CONFIGS_DIR, _ensure_partition_dir, hf_token

MET_OFFICE_HF_REPO = "openclimatefix/met-office-uk-deterministic-solar"

# our column  ->  (Met Office group suffix, inner zarr variable name)
_MET_VARMAP = {
    "ssrd": ("radiation_flux_in_shortwave_total_downward_at_surface",
             "surface_downwelling_shortwave_flux_in_air"),
    "tcc":  ("cloud_amount_of_total_cloud", "cloud_area_fraction"),
    "lcc":  ("cloud_amount_of_low_cloud", "low_type_cloud_area_fraction"),
    "t2m":  ("temperature_at_screen_level", "air_temperature"),
    "ws10": ("wind_speed_at_10m", "wind_speed"),
}
_MET_SUFFIX2COL = {grp: col for col, (grp, _inner) in _MET_VARMAP.items()}

_MET_CRS_CF = {
    "grid_mapping_name": "lambert_azimuthal_equal_area",
    "latitude_of_projection_origin": 54.9,
    "longitude_of_projection_origin": -2.5,
    "false_easting": 0.0,
    "false_northing": 0.0,
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.314140356,
}

_MET_DEFAULT_POINTS = [
    ("South_East", 51.20, 0.50), ("London", 51.51, -0.13),
    ("East_Anglia", 52.40, 0.90), ("South_West", 50.80, -3.50),
    ("Midlands", 52.48, -1.90), ("North", 54.00, -1.50),
    ("Scotland", 56.50, -4.20),
]


def _load_uk_points() -> list[tuple[str, float, float]]:
    """Frozen extraction points from configs/uk_weather_points.csv (region,lat,lon)."""
    cfg = CONFIGS_DIR / "uk_weather_points.csv"
    if not cfg.exists():
        logger.warning(f"{cfg} not found; using built-in default UK points")
        return _MET_DEFAULT_POINTS
    df = pd.read_csv(cfg)
    return [(str(r.region), float(r.lat), float(r.lon)) for r in df.itertuples()]


def _met_office_targets(
    api, years: list[int], months: list[int] | None, hours: list[int] | None
) -> list[str]:
    """List repo .zarr.zip files matching the requested years/months/init-hours."""
    import re

    want_years = {int(y) for y in years}
    want_months = {int(m) for m in months} if months else None
    want_hours = {int(h) for h in hours} if hours else None

    pat = re.compile(r"data/(\d{4})/(\d{2})/(\d{2})/\d{4}-\d{2}-\d{2}-(\d{2})\.zarr\.zip$")
    targets: list[str] = []
    for f in api.list_repo_files(MET_OFFICE_HF_REPO, repo_type="dataset"):
        m = pat.match(f)
        if not m:
            continue
        yr, mo, _, hh = (int(g) for g in m.groups())
        if yr not in want_years:
            continue
        if want_months is not None and mo not in want_months:
            continue
        if want_hours is not None and hh not in want_hours:
            continue
        targets.append(f)
    return sorted(targets)


def _met_out_path(repo_file: str) -> Path:
    stem = Path(repo_file).stem.replace(".zarr", "")  
    y, mo, d, hh = stem.split("-")
    out_dir = _ensure_partition_dir("met_office_nwp", int(y), int(mo))
    return out_dir / f"nwp_{y}{mo}{d}-{hh}Z.parquet"


def _extract_points_from_zip(local_zip: str, repo_file: str,
                             points: list[tuple[str, float, float]]) -> pd.DataFrame:
    import re
    import numpy as np
    import zarr
    from pyproj import CRS, Transformer

    tf = Transformer.from_crs("EPSG:4326", CRS.from_cf(_MET_CRS_CF), always_xy=True)
    store = zarr.storage.ZipStore(local_zip, mode="r")
    try:
        zg = zarr.open_group(store, mode="r")
        group_names = list(zg.group_keys())

        any_grp = group_names[0]
        xc = np.asarray(zg[any_grp]["projection_x_coordinate"][:])
        yc = np.asarray(zg[any_grp]["projection_y_coordinate"][:])
        ys, xs = [], []
        for _region, lat, lon in points:
            gx, gy = tf.transform(lon, lat)
            xs.append(int(np.abs(xc - gx).argmin()))
            ys.append(int(np.abs(yc - gy).argmin()))
        ys = np.asarray(ys); xs = np.asarray(xs)

        init_stem = Path(repo_file).stem.replace(".zarr", "")  # 2023-01-01-00
        init_time = pd.Timestamp(f"{init_stem[:10]}T{init_stem[11:]}:00", tz="UTC")

        pat = re.compile(r"^(\d{8}T\d{4}Z)-PT(\d{4})H\d{2}M-(.+)\.zarr$")
        rows: dict[tuple, dict] = {}
        for g in group_names:
            m = pat.match(g)
            if not m:
                continue
            valid_s, lead, suffix = m.group(1), int(m.group(2)), m.group(3)
            col = _MET_SUFFIX2COL.get(suffix)
            if col is None:
                continue  
            _grp, inner = _MET_VARMAP[col]
            vals = np.asarray(zg[g][inner].get_coordinate_selection((ys, xs)))
            valid_time = pd.Timestamp(valid_s, tz="UTC")  
            for i, (region, _lat, _lon) in enumerate(points):
                rows.setdefault((valid_time, lead, region), {})[col] = float(vals[i])
    finally:
        store.close()

    region_xy = {r: (la, lo) for r, la, lo in points}
    out = []
    for (valid_time, lead, region), vals in rows.items():
        la, lo = region_xy[region]
        out.append({
            "init_time": init_time, "valid_time": valid_time, "forecast_hour": lead,
            "region": region, "lat": la, "lon": lo,
            "ssrd": vals.get("ssrd"), "tcc": vals.get("tcc"), "lcc": vals.get("lcc"),
            "t2m": vals.get("t2m"), "ws10": vals.get("ws10"),
        })
    df = pd.DataFrame(out).sort_values(["valid_time", "region"]).reset_index(drop=True)
    df["source_file"] = repo_file
    df["extracted_at"] = datetime.now(timezone.utc).isoformat()
    return df


def _extract_one(repo_file: str, points: list[tuple[str, float, float]],
                 overwrite: bool) -> tuple[str, str]:
    """Download one zip to temp, point-extract -> parquet, delete temp. Returns status"""
    import shutil
    import tempfile
    from huggingface_hub import hf_hub_download

    out_path = _met_out_path(repo_file)
    if not overwrite and out_path.exists() and out_path.stat().st_size > 256:
        return repo_file, "skip"

    tmp = tempfile.mkdtemp(prefix="metnwp_")
    try:
        local = hf_hub_download(
            repo_id=MET_OFFICE_HF_REPO, filename=repo_file, repo_type="dataset",
            token=hf_token, local_dir=tmp,
        )
        df = _extract_points_from_zip(local, repo_file, points)
        if df.empty:
            return repo_file, "empty"
        df.to_parquet(out_path, index=False, compression="snappy")
        return repo_file, "done"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  


def ingest_met_office_nwp(
    years: list[int],
    months: list[int] | None = None,
    hours: list[int] | None = (0, 12),
    workers: int = 4,
    overwrite: bool = False,
) -> None:

    try:
        from huggingface_hub import HfApi
        import pyproj  
        import zarr  
    except ImportError as e:
        logger.error(f"missing dep: {e}. pip install huggingface_hub pyproj zarr xarray pyarrow")
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api = HfApi(token=hf_token)
    points = _load_uk_points()
    hours_label = "ALL (24/day)" if hours is None else ",".join(f"{h:02d}Z" for h in sorted(hours))
    logger.info(
        f"Met Office THIN bronze | years={years} months={months or 'all'} "
        f"init-hours={hours_label} points={len(points)} workers={workers}"
    )

    targets = _met_office_targets(api, years, months, hours)
    if not targets:
        logger.warning("No matching .zarr.zip files for the given filters")
        return
    logger.info(f"{len(targets):,} init-times selected (~{len(targets) * 12 / 1024:.1f} MB output)")

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one, f, points, overwrite): f for f in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            try:
                _, status = fut.result()
                if status == "done":
                    done += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    logger.warning(f"  {status} {f}")
            except Exception as e:
                failed += 1
                logger.error(f"  failed {f}: {e}")
            if i % 25 == 0 or i == len(targets):
                logger.info(f"progress {i}/{len(targets)} (done={done} skip={skipped} fail={failed})")

    logger.success(
        f"Met Office thin bronze complete: extracted={done} skipped={skipped} failed={failed} "
        f"-> {BRONZE_LOCAL_DIR / 'met_office_nwp'}"
    )


def mirror_met_office_nwp_raw(
    years: list[int],
    months: list[int] | None = None,
    hours: list[int] | None = (0,),
    workers: int = 4,
    overwrite: bool = False,
) -> None:
    """Optional: mirror RAW .zarr.zip files (e.g. 1 backup year). ~143 MB each."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        logger.error("huggingface_hub required: pip install huggingface_hub")
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api = HfApi(token=hf_token)
    out_dir = BRONZE_LOCAL_DIR / "met_office_nwp_raw"
    targets = _met_office_targets(api, years, months, hours)
    if not targets:
        logger.warning("No matching .zarr.zip files for the given filters")
        return
    logger.warning(
        f"RAW mirror of {len(targets):,} files (~{len(targets) * 0.143:.0f} GB) -> {out_dir}"
    )

    def _one(rf: str) -> str:
        lp = out_dir / rf
        if not overwrite and lp.exists() and lp.stat().st_size > 1024:
            return "skip"
        hf_hub_download(repo_id=MET_OFFICE_HF_REPO, filename=rf, repo_type="dataset",
                        token=hf_token, local_dir=str(out_dir))
        return "done"

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, f): f for f in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                status = fut.result()
                done += status == "done"
                skipped += status == "skip"
            except Exception as e:
                failed += 1
                logger.error(f"  failed {futures[fut]}: {e}")
            if i % 25 == 0 or i == len(targets):
                logger.info(f"raw progress {i}/{len(targets)} (new={done} skip={skipped} fail={failed})")
    logger.success(f"Raw mirror complete: downloaded={done} skipped={skipped} failed={failed}")

"""PV_Live GSP observations (api.pvlive.uk) -> Bronze parquet"""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
from loguru import logger
from .common import _ensure_partition_dir, _is_partition_done

def ingest_pv_live(years: list[int]) -> None:
    import httpx

    BASE_URL = "https://api.pvlive.uk/pvlive/api/v4"
    for year in years:
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01T00:00:00Z"
            if month == 12:
                end = f"{year + 1}-01-01T00:00:00Z"
            else:
                end = f"{year}-{month+1:02d}-01T00:00:00Z"

            out_dir = _ensure_partition_dir("pv_live", year, month)
            out_path = out_dir / "gsp_observations.parquet"
            if _is_partition_done(out_path):
                logger.info(f"skip {out_path} (already exists)")
                continue

            logger.info(f"Fetching PV_Live {year}-{month:02d}")

            try:
                r = httpx.get(
                    f"{BASE_URL}/gsp/0",
                    params={"start": start, "end": end, "extra_fields": "capacity_mwp"},
                    timeout=60,
                )
                r.raise_for_status()
                payload = r.json()
                rows = payload.get("data", [])
                if not rows:
                    logger.warning(f"empty payload for {year}-{month:02d}")
                    continue
                cols = payload.get("meta", []) or ["gsp_id", "datetime_gmt", "generation_mw", "capacity_mwp"]
                df = pd.DataFrame(rows, columns=cols)
                df["fetched_at"] = datetime.now(timezone.utc).isoformat()
                df.to_parquet(out_path, index=False, compression="snappy")
                logger.success(f"wrote {len(df):,} rows -> {out_path}")
            except Exception as e:
                logger.error(f"  PV_Live fetch failed for {year}-{month:02d}: {e}")

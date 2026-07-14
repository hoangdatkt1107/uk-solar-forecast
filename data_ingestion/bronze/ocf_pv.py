"""OCF UK PV (HuggingFace openclimatefix/uk_pv) -> Bronze parquet (streamed)"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from .common import _ensure_partition_dir, hf_token

_OCF_CHUNK = 2_000_000
_OCF_FLUSH = 4_000_000


def _flush_ocf(key: tuple[int, int], rows: list) -> None:
    year, month = key
    out_dir = _ensure_partition_dir("ocf_pv", year, month)
    unique_id = uuid.uuid4().hex[:8]
    out_path = out_dir / f"{unique_id}.parquet"
    df = pd.DataFrame(rows)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.success(f"wrote {len(df):,} row -> {out_path}")

def _process_ocf_chunk(chunk: list, buffers: dict[tuple[int, int], list]) -> None:
    df = pd.DataFrame(chunk)
    df["_ts"] = pd.to_datetime(df["datetime_GMT"], utc=True, errors="coerce")
    df["_year"] = df["_ts"].dt.year
    df["_month"] = df["_ts"].dt.month
    for (yr, mo), grp in df.groupby(["_year", "_month"], sort=False):
        key = (int(yr), int(mo))
        rows = grp.drop(columns=["_ts", "_year", "_month"]).to_dict("records")
        buffers.setdefault(key, []).extend(rows)

def ingest_ocf(years: list[int]) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package required: pip install datasets")
        return

    logger.info("Streaming OCF UK PV")

    for year in years:
        for month in range(1, 13):
            target_file = f"30_minutely/year={year}/month={month:02d}/*.parquet"
            logger.info(f"collecting data from partition year {year}, month {month:02d}")

            try:
                ds = load_dataset(
                    "openclimatefix/uk_pv",
                    split="train",
                    streaming=True,
                    data_files=target_file,
                    token=hf_token,
                )
            except Exception as e:
                logger.warning(f"skip the monthly data {month:02d}/{year}: {e}")
                continue

            buffers: dict[tuple[int, int], list] = {}

            for batch in ds.iter(batch_size=_OCF_CHUNK):
                _process_ocf_chunk(batch, buffers)

                for key in list(buffers):
                    if len(buffers[key]) >= _OCF_FLUSH:
                        _flush_ocf(key, buffers[key])
                        buffers[key] = []
            for key, rows in buffers.items():
                if rows:
                    _flush_ocf(key, rows)

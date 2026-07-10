from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gridsight.config import settings

# reuse the Silver reader so Gold always consumes LOCAL silver parquet
from ..silver.common import SILVER_LOCAL_DIR, read_silver 

GOLD_LOCAL_DIR = settings.data_dir / "gold"
GOLD_TABLE = "gold_features"
hf_token = settings.hf_token
GOLD_HF_REPO = (getattr(settings, "gold_hf_repo", None)
                or settings.bronze_hf_repo.replace("bronze", "gold"))

TS = "timestamp_utc"

UK_LAT, UK_LON = 54.0, -2.5


def write_gold(df: pd.DataFrame, table: str = GOLD_TABLE) -> int:
    """Write the Gold table partitioned by year=YYYY/month=MM (clean rebuild)."""
    import shutil
    if df.empty:
        logger.warning(f"{table}: nothing to write")
        return 0
    shutil.rmtree(GOLD_LOCAL_DIR / table, ignore_errors=True)
    df = df.sort_values(TS).reset_index(drop=True)
    yr, mo = df[TS].dt.year, df[TS].dt.month
    written = 0
    for (y, m), part in df.groupby([yr, mo]):
        out_dir = GOLD_LOCAL_DIR / table / f"year={int(y)}" / f"month={int(m):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out_dir / f"{table}_{int(y)}{int(m):02d}.parquet",
                        index=False, compression="snappy")
        written += len(part)
    logger.success(f"{table}: wrote {written:,} rows x {df.shape[1]} cols -> {GOLD_LOCAL_DIR / table}")
    return written


def read_gold(table: str = GOLD_TABLE) -> pd.DataFrame:
    files = sorted((GOLD_LOCAL_DIR / table).rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

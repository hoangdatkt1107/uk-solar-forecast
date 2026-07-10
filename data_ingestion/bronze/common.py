"""Shared config & helpers for the Bronze layer ingestion package"""
from __future__ import annotations
import sys
from pathlib import Path

# make `gridsight` (under <project>/src) importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gridsight.config import settings

hf_token = settings.hf_token
BRONZE_LOCAL_DIR = settings.data_dir / "bronze"
BRONZE_HF_REPO = settings.bronze_hf_repo
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _ensure_partition_dir(source: str, year: int, month: int, day: int | None = None) -> Path:
    parts = [f"year={year}", f"month={month:02d}"]
    if day is not None:
        parts.append(f"day={day:02d}")
    path = BRONZE_LOCAL_DIR / source / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_partition_done(out_path: Path) -> bool:
    check_if_valid_file = out_path.exists() and out_path.stat().st_size > 1024
    return check_if_valid_file

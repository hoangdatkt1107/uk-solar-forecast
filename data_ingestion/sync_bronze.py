"""Sync Bronze from your HuggingFace dataset repo -> local data/bronze/"""
from __future__ import annotations
import argparse
from loguru import logger
from .bronze.common import BRONZE_LOCAL_DIR, BRONZE_HF_REPO, hf_token

SOURCES = ["ocf_pv", "pv_live", "met_office_nwp", "neso"]

def sync_bronze(source: str = "all") -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub required: pip install huggingface_hub")
        return

    if source == "all":
        patterns = None  # whole repo
    elif source in SOURCES:
        patterns = [f"{source}/**"]
    else:
        logger.error(f"unknown source {source!r} (choices: {SOURCES + ['all']})")
        return

    BRONZE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Syncing hf://datasets/{BRONZE_HF_REPO} ({source}) -> {BRONZE_LOCAL_DIR}")
    snapshot_download(
        repo_id=BRONZE_HF_REPO,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(BRONZE_LOCAL_DIR),
        allow_patterns=patterns,
    )
    logger.success(f"Bronze sync complete ({source}) -> {BRONZE_LOCAL_DIR}")

def main() -> None:
    p = argparse.ArgumentParser(description="Sync Bronze from HF to local data/bronze/")
    p.add_argument("--source", default="all", choices=SOURCES + ["all"])
    args = p.parse_args()
    sync_bronze(args.source)

if __name__ == "__main__":
    main()

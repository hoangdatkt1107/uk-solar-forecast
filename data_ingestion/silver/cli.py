"""CLI for the Silver layer.

    python -m data_ingestion.silver --source all
    python -m data_ingestion.silver --source nwp

Silver rebuilds entirely from LOCAL Bronze parquet (no network).
"""
from __future__ import annotations
import argparse

from loguru import logger

from .pv_live import run as run_pv_live
from .ocf_pv import run as run_ocf_pv
from .met_office import run as run_met_office
from .neso import run as run_neso
from .contracts import cross_source_checks
from .upload import upload_silver_to_hf

# CLI source name -> output table name (for upload)
_TABLE = {
    "ocf_pv": "silver_ocf_pv",
    "pv_live": "silver_pv_live",
    "met_office_nwp": "silver_met_office_nwp",
    "neso": "silver_neso",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Silver layer build (from local Bronze)")
    p.add_argument(
        "--source", required=True,
        choices=["ocf_pv", "pv_live", "met_office_nwp", "neso", "all", "cross_check"],
    )
    p.add_argument(
        "--no-cross-check", action="store_true",
        help="skip cross-source sanity checks in 'all' mode",
    )
    p.add_argument(
        "--upload", action="store_true",
        help="upload built Silver table(s) to your HF silver repo after building",
    )
    args = p.parse_args()

    handlers = {
        "ocf_pv": run_ocf_pv,
        "pv_live": run_pv_live,
        "met_office_nwp": run_met_office,
        "neso": run_neso,
    }

    if args.source == "cross_check":
        logger.info("=== Cross-source checks ===")
        cross_source_checks()
        return
    sources = list(handlers) if args.source == "all" else [args.source]

    for src in sources:
        logger.info(f"=== Building silver_{src} ===")
        handlers[src]()
        if args.upload:
            upload_silver_to_hf(_TABLE[src])


if __name__ == "__main__":
    main()

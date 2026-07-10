"""CLI entrypoint for the Bronze layer ingestion."""
from __future__ import annotations
import argparse
from datetime import datetime

import pandas as pd
from loguru import logger

from .pv_live import ingest_pv_live
from .ocf_pv import ingest_ocf
from .met_office import ingest_met_office_nwp, mirror_met_office_nwp_raw
from .neso import ingest_neso
from .upload import upload_to_hf


def main() -> None:
    p = argparse.ArgumentParser(description="Bronze layer ingestion")
    p.add_argument(
        "--source",
        required=True,
        choices=["ocf_pv", "pv_live", "met_office_nwp", "neso", "all"],
    )
    p.add_argument("--years", type=int, nargs="+", default=[datetime.now().year])
    p.add_argument(
        "--date", type=str, default=None,
        help="ISO date for single-day ingest (overrides --years for HF sources)",
    )
    p.add_argument(
        "--months", type=int, nargs="+", default=None,
        help="met_office_nwp: months to ingest (1-12). Default: all months in --years",
    )
    p.add_argument(
        "--hours", type=int, nargs="+", default=[0, 12],
        help="met_office_nwp: init-times/day to keep (0-23). "
             "Default 0 12. Use '0' for 1 run/day, or '--hours -1' for all 24.",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="met_office_nwp: parallel download/extract threads (default 4)",
    )
    p.add_argument(
        "--raw", action="store_true",
        help="met_office_nwp: mirror RAW .zarr.zip (~143 MB each) instead of thin "
             "point-extraction. Use only for a small backup set.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="met_office_nwp: re-extract/re-download even if output already exists",
    )
    p.add_argument(
        "--packages", nargs="+", default=None,
        help="neso: CKAN package ids (default: GridSight-relevant set)",
    )
    p.add_argument(
        "--force-refresh", action="store_true",
        help="neso: ignore last_modified cache and re-fetch every resource",
    )
    p.add_argument(
        "--max-records", type=int, default=None,
        help="neso: cap records per resource (useful for dev / testing)",
    )
    p.add_argument(
        "--upload", action="store_true",
        help="Upload local Bronze to HF after ingestion",
    )
    args = p.parse_args()

    if args.date:
        dt = pd.Timestamp(args.date) if args.date != "today" else pd.Timestamp.now()
        args.years = [dt.year]
        args.months = [dt.month]

    # --hours -1 is the escape hatch for "keep all 24 init-times/day"
    met_hours = None if args.hours and -1 in args.hours else args.hours
    met_fn = mirror_met_office_nwp_raw if args.raw else ingest_met_office_nwp

    handlers = {
        "ocf_pv": lambda: ingest_ocf(args.years),
        "pv_live": lambda: ingest_pv_live(args.years),
        "met_office_nwp": lambda: met_fn(
            args.years, months=args.months, hours=met_hours,
            workers=args.workers, overwrite=args.overwrite,
        ),
        "neso": lambda: ingest_neso(
            packages=args.packages, years=args.years,
            force_refresh=args.force_refresh, max_records=args.max_records,
        ),
    }
    sources = list(handlers.keys()) if args.source == "all" else [args.source]

    for src in sources:
        logger.info(f"=== Ingesting {src} ===")
        handlers[src]()
        if args.upload:
            upload_to_hf(src)


if __name__ == "__main__":
    main()

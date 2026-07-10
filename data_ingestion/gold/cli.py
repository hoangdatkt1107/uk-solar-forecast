"""How to use (please see the command below)
    python -m data_ingestion.gold                       # day-ahead (horizon=48)
    python -m data_ingestion.gold --horizon-steps 6     # 3h-ahead
    python -m data_ingestion.gold --upload              # build + push to HF
"""
from __future__ import annotations
import argparse

from loguru import logger

from .build import run, DEFAULT_HORIZON


def main() -> None:
    p = argparse.ArgumentParser(description="Gold feature-store build (from local Silver)")
    p.add_argument(
        "--horizon-steps", type=int, default=DEFAULT_HORIZON,
        help=f"forecast horizon in 30-min steps (default {DEFAULT_HORIZON}=24h day-ahead). "
             "Observed-actual lags are forced >= this to stay leakage-free.",
    )
    p.add_argument("--upload", action="store_true",
                   help="upload Gold to your HF gold repo after building")
    args = p.parse_args()

    logger.info(f"=== Building Gold (horizon={args.horizon_steps} steps) ===")
    run(horizon=args.horizon_steps, upload=args.upload)


if __name__ == "__main__":
    main()

"""Cloud pipeline entrypoint for the Container Apps jobs
1) python pipeline.py serve      # hourly
2) python pipeline.py retrain    # weekly

Data flow, env knobs and gotchas: docs/cloud_pipeline.md
"""
from __future__ import annotations
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from loguru import logger


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _horizons() -> list[int]:
    raw = os.getenv("GRIDSIGHT_HORIZONS", "24,12")
    out = [int(t) for t in (t.strip() for t in raw.split(",")) if t]
    return out or [24, 12]


def _on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip() in ("1", "true", "True")

def refresh_bronze() -> None:
    """Sync bronze from HF, then top up the recent tail from the live sources."""
    from data_ingestion.bronze.common import BRONZE_LOCAL_DIR
    from data_ingestion.bronze.met_office_aws import ingest_met_office_aws
    from data_ingestion.bronze.pv_live import ingest_pv_live
    from data_ingestion.bronze.neso import ingest_neso

    now = datetime.now(timezone.utc)
    lookback = _env_int("GRIDSIGHT_BRONZE_LOOKBACK_DAYS", 7)
    overwrite_days = _env_int("GRIDSIGHT_BRONZE_OVERWRITE_DAYS", 1)
    # LEAN (serve): the NESO archive is ~38M rows across all years. Pulling + rebuilding all
    # of it OOMs the small container, and serve only needs the current year. Weekly retrain
    # leaves this off and does the full history.
    neso_lean = _on("GRIDSIGHT_NESO_LEAN", "0")

    if not _on("GRIDSIGHT_SKIP_HF_SYNC", "0"):
        try:
            from data_ingestion.bronze.common import BRONZE_HF_REPO, hf_token
            from huggingface_hub import snapshot_download
            months = _env_int("GRIDSIGHT_SYNC_MONTHS", 2)     # rolling window, not the full 2yr
            if neso_lean:                                     # only this year's neso archive
                pats = [f"neso/**/*{now.year}*",
                        "neso/**/embedded_solar_and_wind_forecast.parquet"]
            else:
                pats = ["neso/**"]                            # neso isn't year/month partitioned
            d = now.replace(day=1)
            for _ in range(months):
                pats.append(f"*/year={d.year}/month={d.month:02d}/**")
                d = (d - timedelta(days=1)).replace(day=1)
            logger.info(f"refresh: syncing last {months} months of bronze from HuggingFace")
            BRONZE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=BRONZE_HF_REPO, repo_type="dataset", token=hf_token,
                              local_dir=str(BRONZE_LOCAL_DIR), allow_patterns=pats)
        except Exception as e:
            logger.warning(f"refresh: HF sync failed ({e}); continuing with local bronze")

    # Met Office NWP: gap-fill the window, then overwrite the last day so late-published
    # leads get filled in
    gap_start = (now - timedelta(days=lookback)).strftime("%Y-%m-%d")
    ow_start = (now - timedelta(days=overwrite_days)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    met_workers = _env_int("GRIDSIGHT_MET_WORKERS", 2)   # low: many concurrent HDF5 opens OOM/crash the small container
    logger.info(f"refresh: met AWS gap-fill {gap_start}->now (workers={met_workers})")
    ingest_met_office_aws(start=gap_start, end=end, overwrite=False, workers=met_workers)
    # The overwrite pass re-fetches a whole day; a single corrupt S3 download crashes the
    # netCDF/HDF5 native lib and kills the job. Skip it in the cloud (sync + gap-fill cover it).
    if _on("GRIDSIGHT_MET_OVERWRITE", "1"):
        logger.info(f"refresh: met AWS overwrite last {overwrite_days}d")
        ingest_met_office_aws(start=ow_start, end=end, overwrite=True, workers=met_workers)

    # PV_Live skips months it already has, so drop the recent ones to force a re-fetch
    months = {(now.year, now.month)}
    prev = now.replace(day=1) - timedelta(days=1)
    months.add((prev.year, prev.month))
    for y, m in months:
        shutil.rmtree(BRONZE_LOCAL_DIR / "pv_live" / f"year={y}" / f"month={m:02d}", ignore_errors=True)
    logger.info(f"refresh: pv_live re-fetch months {sorted(months)}")
    ingest_pv_live(years=sorted({y for y, _ in months}))

    logger.info("refresh: neso embedded forecast")
    ingest_neso(force_refresh=True, years=[now.year] if neso_lean else None)

def push_bronze() -> None:
    """Push fresh bronze back to HF. Required every run — old NWP is purged from S3."""
    if not _on("GRIDSIGHT_PUSH_BRONZE"):
        logger.info("push_bronze: disabled")
        return
    from data_ingestion.bronze.upload import upload_to_hf
    for src in ("met_office_nwp", "pv_live", "neso"):
        try:
            upload_to_hf(src)
        except Exception as e:
            logger.warning(f"push_bronze[{src}]: failed ({e})")

def push_silver_gold() -> None:
    """Sync silver + gold to HF. Retrain only — hourly would re-upload the full rebuild."""
    if not _on("GRIDSIGHT_PUSH_DERIVED"):
        logger.info("push_silver_gold: disabled")
        return
    try:
        from data_ingestion.silver.upload import upload_silver_to_hf
        upload_silver_to_hf()
    except Exception as e:
        logger.warning(f"push_silver: failed ({e})")
    try:
        from data_ingestion.gold.upload import upload_gold_to_hf
        upload_gold_to_hf()
    except Exception as e:
        logger.warning(f"push_gold: failed ({e})")

def build_silver() -> None:
    from data_ingestion.silver.pv_live import run as sv_pv_live
    from data_ingestion.silver.met_office import run as sv_met
    from data_ingestion.silver.neso import run as sv_neso

    logger.info("silver: building pv_live / met_office_nwp / neso")
    sv_pv_live()
    sv_met()
    sv_neso()

def serve() -> None:
    import dataclasses
    from modeling.config import ModelConfig
    from modeling.serve import serve_all

    refresh_bronze()
    push_bronze()
    build_silver()

    out_dir = os.getenv("GRIDSIGHT_SERVE_DIR", "artifacts/serve")
    models = tuple(m.strip() for m in os.getenv("GRIDSIGHT_MODELS", "stack,chronos").split(",") if m.strip())
    for h in _horizons():
        cfg = dataclasses.replace(ModelConfig(), horizon_step=h)
        logger.info(f"serve: horizon={h} steps ({h // 2}h) models={models} -> {out_dir}")
        serve_all(cfg, models=models)
    logger.success("serve cycle complete")


def retrain() -> None:
    import dataclasses
    from data_ingestion.gold.build import run as gold_run
    from modeling.config import ModelConfig
    from modeling.train import run as train_run
    from modeling.registry import push_model_if_better

    refresh_bronze()
    push_bronze()
    build_silver()
    for h in _horizons():
        logger.info(f"retrain: building gold + training horizon={h} steps ({h // 2}h)")
        gold_run(horizon=h)
        cfg = dataclasses.replace(ModelConfig(), horizon_step=h)
        train_run(cfg)
        push_model_if_better(cfg)      # promote only if it beats the live model
    push_silver_gold()
    logger.success("retrain complete")


_COMMANDS = {"serve": serve, "retrain": retrain, "refresh": refresh_bronze,
             "silver": build_silver}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    fn = _COMMANDS.get(cmd)
    if fn is None:
        logger.error(f"unknown command {cmd!r}; choose from {sorted(_COMMANDS)}")
        return 2
    fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

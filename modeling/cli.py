"""CLI for the probabilistic solar-forecasting stack.
    python -m modeling.train_cli                (full train + evaluate)
    python -m modeling.train_cli --fast          quick smoke run (few epochs)
    python -m modeling.train_cli --target target_mw --no-daylight
"""
from __future__ import annotations
import argparse
import dataclasses

from loguru import logger

from .config import ModelConfig
from .train import run


def _build_cfg(a) -> ModelConfig:
    cfg = ModelConfig()
    if a.gold_dir:
        from pathlib import Path
        cfg = dataclasses.replace(cfg, gold_dir=Path(a.gold_dir))
    cfg = dataclasses.replace(
        cfg,
        target=a.target, horizon_steps=a.horizon_steps, n_folds=a.n_folds,
        seq_len=a.seq_len, tcn_epochs=a.epochs,
        val_start=a.val_start, test_start=a.test_start,
        daylight_only=not a.no_daylight,
    )
    if a.fast:  # quick smoke config
        cfg = dataclasses.replace(
            cfg, tcn_epochs=2, n_folds=3, seq_len=48,
            tcn_channels=(32, 32, 32, 32),
            lgbm_params={**cfg.lgbm_params, "n_estimators": 120},
        )
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Train TCN-Q + LGBM-Q -> Linear-Q stack")
    p.add_argument("--target", default="target_cf", choices=["target_cf", "target_mw"])
    p.add_argument("--horizon-steps", type=int, default=48)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=126)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--val-start", default="2024-07-01")
    p.add_argument("--test-start", default="2024-10-01")
    p.add_argument("--no-daylight", action="store_true", help="score all hours, not just daylight")
    p.add_argument("--gold-dir", default=None)
    p.add_argument("--fast", action="store_true", help="quick smoke run")
    args = p.parse_args()

    cfg = _build_cfg(args)
    logger.info(f"config: target={cfg.target} seq_len={cfg.seq_len} folds={cfg.n_folds} "
                f"epochs={cfg.tcn_epochs} daylight={cfg.daylight_only}")
    run(cfg)


if __name__ == "__main__":
    main()

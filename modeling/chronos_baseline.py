"""
Chronos sees only the target's own history (no NWP), so it does not depend on
the weather ingest. Run standalone: python -m modeling.chronos_baseline"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from loguru import logger
from .config import ModelConfig
from data_ingestion.gold.common import read_gold, gold_table, TS
from . import metrics


def _load_target(cfg: ModelConfig, horizon: int):
    df = read_gold(gold_table(horizon)).sort_values(TS).reset_index(drop=True)
    y = df[cfg.target].to_numpy("float32")
    ts = df[TS]
    if "is_daylight" in df:
        daylight = (df["is_daylight"] == 1).to_numpy()
    elif "solar_elevation_deg" in df:
        daylight = (df["solar_elevation_deg"] > 0).to_numpy()
    else:
        daylight = np.ones(len(df), dtype=bool)
    test = (ts >= pd.Timestamp(cfg.test_start, tz="UTC")).to_numpy()
    cap = df["capacity_mwp"].to_numpy("float32") if "capacity_mwp" in df else None
    base = df["embedded_solar_mw"].to_numpy("float32") if "embedded_solar_mw" in df else None
    return y, daylight, test, cap, base


def run(cfg: ModelConfig = ModelConfig(), horizon: int = 24,
        model_name: str = "amazon/chronos-bolt-base", context_len: int = 512,
        stride: int = 1, max_eval: int | None = None, device: str | None = None) -> dict:
    from chronos import BaseChronosPipeline
    qs = list(cfg.quantiles)
    y, daylight, test, cap, base = _load_target(cfg, horizon)

    # predict each scored test slot t from origin t-horizon (leakage-safe, fixed lead)
    idx = np.where(test & daylight)[0]
    idx = idx[idx - horizon - context_len >= 0]
    idx = idx[np.isfinite(y[idx])]                 
    if stride > 1:
        idx = idx[::stride]
    if max_eval:
        idx = idx[:max_eval]
    if not len(idx):
        logger.error("no eval slots (need test daylight rows with enough history)")
        return {}

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Chronos {model_name} | horizon={horizon} ctx={context_len} "
                f"eval_slots={len(idx)} device={device}")
    pipe = BaseChronosPipeline.from_pretrained(model_name, device_map=device,
                                               torch_dtype=torch.float32)

    preds = {q: np.empty(len(idx), "float32") for q in qs}
    batch = 256
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        ctx = [torch.tensor(np.nan_to_num(y[i - horizon - context_len + 1: i - horizon + 1]))
               for i in chunk]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=horizon,
                                      quantile_levels=qs)
        last = q[:, -1, :].cpu().numpy()          # step landing on t = origin + horizon
        for j, ql in enumerate(qs):
            preds[ql][s:s + len(chunk)] = last[:, j]
        logger.info(f"  {min(s + batch, len(idx))}/{len(idx)}")

    yt = y[idx]
    rep = metrics.report(yt, preds, tuple(qs))
    if base is not None and cap is not None:
        base_cf = base[idx] / np.clip(cap[idx], 1e-6, None)
        rep["skill_vs_neso_q50"] = metrics.skill_vs_baseline(yt, preds[0.5], base_cf)
    logger.success("Chronos baseline: " + " ".join(f"{k}={v:.4f}" for k, v in rep.items()))
    return rep


if __name__ == "__main__":
    run()

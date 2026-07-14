from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ModelConfig
from .data import make_sequences
from .clearsky import clearsky_feature
from .base import TCNQuantile
from .stacking import assemble_meta_X

def load_stack(artifacts_dir: str | Path):
    import joblib
    import torch
    from .registry import pull_model_dir
    d = pull_model_dir(artifacts_dir)                 # HF live model, or baked fallback
    art = joblib.load(Path(d) / "stack.joblib")
    cfg = art["cfg"]
    tcn = TCNQuantile(cfg, len(art["features"])).build()
    tcn.model_.load_state_dict(torch.load(Path(d) / "tcn.pt", map_location="cpu"))
    tcn.model_.eval()
    return art, tcn

def predict_gold(df: pd.DataFrame, artifacts_dir: str | Path = "artifacts/model") -> pd.DataFrame:
    art, tcn = load_stack(artifacts_dir)
    cfg: ModelConfig = art["cfg"]
    feats, std, lgbm, meta = art["features"], art["standardizer"], art["lgbm"], art["meta"]
    qnames = cfg.quantile_names()

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    V = df[feats].to_numpy("float32")
    seqs, end_idx = make_sequences(std.transform(V), cfg.seq_len)
    clear = clearsky_feature(df)

    lgp = lgbm.predict(V[end_idx])
    tcp = tcn.predict(seqs)
    preds = meta.predict(assemble_meta_X(tcp, lgp, clear[end_idx], cfg.quantiles))

    out = df.iloc[end_idx][["timestamp_utc"]].copy()
    cap = df["capacity_mwp"].to_numpy("float32")[end_idx] if "capacity_mwp" in df else 1.0
    day = df["is_daylight"].to_numpy()[end_idx] if "is_daylight" in df else np.ones(len(end_idx))
    for q, name in zip(cfg.quantiles, qnames):
        p = preds[q] * cap if cfg.target == "target_cf" else preds[q]
        out[f"pred_{name}"] = np.where(day == 1, p, 0.0).astype("float32")   # night -> 0
    return out.reset_index(drop=True)
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .config import ModelConfig
from .data import prepare, make_sequences, Standardizer
from .clearsky import clearsky_feature
from .base.lgbm_q import LGBMQuantile
from .base.tcn_q import TCNQuantile
from .stacking import LinearQuantileStacker, assemble_meta_X
from . import metrics
import pdb

def run(cfg: ModelConfig) -> dict:
    np.random.seed(cfg.seed)
    ds = prepare(cfg)
    qs = cfg.quantiles
    df = ds.df
    V = df[ds.feature_columns].to_numpy("float32")
    y = df[cfg.target].to_numpy("float32")
    clear = clearsky_feature(df)
    cap = df["capacity_mwp"].to_numpy("float32") if "capacity_mwp" in df else None

    tr_mask, va_mask, te_mask = ds.split_masks()
    score = ds.score_mask()
    logger.info(f"rows: train={int((tr_mask & score).sum())} val={int((va_mask & score).sum())} "
                f"test={int((te_mask & score).sum())} | features={len(ds.feature_columns)}")

    std = Standardizer().fit(V[tr_mask])
    Vs = std.transform(V)
    seqs, end_idx = make_sequences(Vs, cfg.seq_len)
    seqpos_of_row = np.full(len(df), -1, dtype="int64")
    seqpos_of_row[end_idx] = np.arange(len(end_idx))

    #-------OUT-OF-FOLD (for Quantile Regressor - examiner)------------
    from sklearn.model_selection import TimeSeriesSplit
    train_rows = np.where(tr_mask & score)
    oof_lgbm = {q: np.full(len(df), np.nan, "float32") for q in qs}
    oof_tcn = {q: np.full(len(df), np.nan, "float32") for q in qs}
    pdb.set_trace()
    return train_rows

test = run(ModelConfig())




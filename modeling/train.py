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

def _tcn_fit_rows(cfg, seqs, seqpos_of_row, rows, y):
    pos = seqpos_of_row[rows]
    ok = pos >= 0
    model = TCNQuantile(cfg, n_features=seqs.shape[2]).fit(seqs[pos[ok]], y[rows][ok])
    return model

def _tcn_predict_rows(model, seqs, seqpos_of_row, rows):
    """Predict only rows that have a full window; returns (preds_dict, ok_mask)."""
    pos = seqpos_of_row[rows]
    ok = pos >= 0
    preds = model.predict(seqs[pos[ok]])
    return preds, ok

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

    #OUT-OF-FOLD (for Quantile Regressor - examiner)
    from sklearn.model_selection import TimeSeriesSplit
    train_rows = np.where(tr_mask & score)[0]
    oof_lgbm = {q: np.full(len(df), np.nan, "float32") for q in qs}
    oof_tcn = {q: np.full(len(df), np.nan, "float32") for q in qs}

    for k, (tr_i, va_i) in enumerate(TimeSeriesSplit(cfg.n_folds).split(train_rows), 1):
        rtr, rva = train_rows[tr_i], train_rows[va_i]
        logger.info(f"OOF fold {k}/{cfg.n_folds}: fit={len(rtr)} predict={len(rva)}")
        lg = LGBMQuantile(qs, cfg.lgbm_params, cfg.seed).fit(V[rtr], y[rtr])
        for q, predict in lg.predict(V[rva]).items():
            oof_lgbm[q][rva] = predict
        tcn = _tcn_fit_rows(cfg, seqs, seqpos_of_row, rtr, y)
        preds, ok = _tcn_predict_rows(tcn, seqs, seqpos_of_row, rva)
        for q in qs:
            oof_tcn[q][rva[ok]] = preds[q]
        
    meta_rows = np.where(tr_mask & score & np.isfinite(oof_tcn[qs[0]]) &
                         np.isfinite(oof_lgbm[qs[0]]))[0]
    
    Z_oof = assemble_meta_X({q: oof_tcn[q][meta_rows] for q in qs},
                            {q: oof_lgbm[q][meta_rows] for q in qs},
                            clear[meta_rows], qs)
    
    meta = LinearQuantileStacker(qs).fit(Z_oof, y[meta_rows])
    logger.success(f"meta-learner fit on {len(meta_rows)} OOF rows")
    
    lg_full = LGBMQuantile(qs, cfg.lgbm_params, cfg.seed).fit(V[train_rows], y[train_rows])
    tcn_full = _tcn_fit_rows(cfg, seqs, seqpos_of_row, train_rows, y)

    def predict_split(mask) -> tuple[dict, np.ndarray]:
        rows = np.where(mask & score)[0]
        pos_ok = seqpos_of_row[rows] >= 0
        rows = rows[pos_ok]                              
        lgp = lg_full.predict(V[rows])
        tcp, _ = _tcn_predict_rows(tcn_full, seqs, seqpos_of_row, rows)
        Z = assemble_meta_X(tcp, lgp, clear[rows], qs)
        return meta.predict(Z), rows

    ts = df["timestamp_utc"].to_numpy()
    results = {}
    for name, mask in [("val", va_mask), ("test", te_mask)]:
        preds, rows = predict_split(mask)
        rep = metrics.report(y[rows], preds, qs)
        base_cf = None
        if "embedded_solar_mw" in df.columns and cap is not None:
            base = df["embedded_solar_mw"].to_numpy("float32")[rows]
            base_cf = base / np.clip(cap[rows], 1e-6, None) if cfg.target == "target_cf" else base
            rep["skill_vs_neso_q50"] = metrics.skill_vs_baseline(y[rows], preds[0.5], base_cf)
        results[name] = rep
        logger.info(f"[{name}] " + " ".join(f"{k}={v:.4f}" for k, v in rep.items()))
        
    _save_artifacts(cfg, lg_full, tcn_full, meta, std, ds.feature_columns, results)
    return results

def _save_predictions(cfg, name, ts, y_true, preds, cap, base):
    """Persist per-slot predictions so `modeling.evaluate` can chart them later."""
    out = pd.DataFrame({"timestamp_utc": ts, "y_true": y_true})
    for q in cfg.quantiles:
        out[f"q{int(q*100)}"] = preds[q]
    if cap is not None:
        out["capacity_mwp"] = cap
    if base is not None:
        out["neso"] = base
    out["target"] = cfg.target
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(cfg.artifacts_dir) / f"pred_{name}.parquet", index=False)

def _save_artifacts(cfg, lg, tcn, meta, std, feat_cols, results):
    out = Path(cfg.artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import joblib, torch
        joblib.dump({"lgbm": lg, "meta": meta, "standardizer": std,
                     "features": feat_cols, "cfg": cfg}, out / "stack.joblib")
        torch.save(tcn.model_.state_dict(), out / "tcn.pt")
    except Exception as e:
        logger.warning(f"artifact save skipped: {e}")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.success(f"artifacts + metrics -> {out}")

if __name__ == "__main__":
    test = run(ModelConfig())




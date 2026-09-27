"""HuggingFace model registry — pull the live model for serving, push a retrained one
only if it holds up against the live one.
Repo layout:
1) model_12h/{stack.joblib, tcn.pt, metrics.json}
2)model_6h/ {stack.joblib, tcn.pt, metrics.json}

Repo id resolves from GRIDSIGHT_MODEL_HF_REPO, else derived from GRIDSIGHT_BRONZE_HF_REPO
(bronze -> model). Serving pulls the latest at load time (falls back to the baked-in
artifacts if HF is unset/unreachable).

Promotion. The split is rolling, so every retrain's test window is newer than the last
one and the metrics.json of two different weeks are not comparable. The gate downloads the
live model and scores it and the new model on the same rows of this run's test window, and
promotes unless the new mean pinball is more than GRIDSIGHT_PROMOTE_TOLERANCE worse
(default 2%: ties and noise go to the model trained on fresher data). A model that does not
beat NESO on its own test window is never promoted.
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
from loguru import logger

def _setting(env_key: str, attr: str) -> str | None:
    """Read a value from the real environment"""
    v = os.getenv(env_key)
    if v:
        return v
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from gridsight.config import settings
        return getattr(settings, attr, None)
    except Exception:
        return None

def _token() -> str | None:
    return _setting("GRIDSIGHT_HF_TOKEN", "hf_token")

def model_repo() -> str | None:
    explicit = _setting("GRIDSIGHT_MODEL_HF_REPO", "model_hf_repo")
    if explicit:
        return explicit
    bronze = _setting("GRIDSIGHT_BRONZE_HF_REPO", "bronze_hf_repo")
    return bronze.replace("bronze", "model") if bronze else None

def pull_model_dir(artifacts_dir: str | Path) -> Path:
    """Return a dir holding the model to serve: the HF copy if available, else the baked
    local `artifacts_dir`. Never raises — serving must not fail on an HF hiccup"""
    baked = Path(artifacts_dir)
    if os.getenv("GRIDSIGHT_MODEL_FROM_HF", "1").strip() not in ("1", "true", "True"):
        return baked
    repo = model_repo()
    if not repo:
        return baked
    tag = baked.name                                  # e.g. "model_12h"
    try:
        from huggingface_hub import snapshot_download
        cache = os.getenv("GRIDSIGHT_MODEL_CACHE", "/tmp/gridsight-models")
        local = snapshot_download(repo_id=repo, repo_type="model",
                                  allow_patterns=[f"{tag}/**"], local_dir=cache,
                                  token=_token())
        d = Path(local) / tag
        if (d / "stack.joblib").exists() and (d / "tcn.pt").exists():
            logger.info(f"model: using HF {repo}/{tag}")
            return d
        logger.warning(f"model: {repo}/{tag} incomplete; using baked {baked}")
    except Exception as e:
        logger.warning(f"model: HF pull failed ({e}); using baked {baked}")
    return baked

def _fetch_live(repo: str, tag: str, token: str | None) -> Path | None:
    """Download the live model into a fresh dir. None if absent or incomplete. Unlike
    pull_model_dir this never falls back to the local artifacts, which after a retrain
    hold the new model, not the live one."""
    from huggingface_hub import snapshot_download
    try:
        local = snapshot_download(repo_id=repo, repo_type="model", allow_patterns=[f"{tag}/**"],
                                  local_dir=tempfile.mkdtemp(prefix="live_model_"), token=token)
    except Exception as e:
        logger.warning(f"push_model[{tag}]: could not download the live model ({e})")
        return None
    d = Path(local) / tag
    return d if (d / "stack.joblib").exists() and (d / "tcn.pt").exists() else None


def _predict_rows(model_dir: Path, df, rows: np.ndarray, clear: np.ndarray, cfg):
    """(rows it could predict, {q: preds}) for the model in model_dir, or None if it was
    trained for another target or quantiles, or needs columns this data does not have.
    Builds sequences only for the requested rows: a full make_sequences over the whole
    history at seq_len=126 is close to 1 GB per model."""
    import joblib
    import torch
    from .base import TCNQuantile
    from .stacking import assemble_meta_X

    art = joblib.load(model_dir / "stack.joblib")
    mcfg, feats = art["cfg"], art["features"]
    if (mcfg.target != cfg.target or tuple(mcfg.quantiles) != tuple(cfg.quantiles)
            or any(f not in df.columns for f in feats)):
        return None
    L = mcfg.seq_len
    rows = rows[rows >= L - 1]                       # same rows train.run can score
    V = df[feats].to_numpy("float32")
    win = np.lib.stride_tricks.sliding_window_view(art["standardizer"].transform(V), L, axis=0)
    seqs = np.ascontiguousarray(win[rows - (L - 1)].transpose(0, 2, 1)).astype("float32")

    tcn = TCNQuantile(mcfg, len(feats)).build()
    tcn.model_.load_state_dict(torch.load(model_dir / "tcn.pt", map_location="cpu"))
    tcn.model_.eval()
    Z = assemble_meta_X(tcn.predict(seqs), art["lgbm"].predict(V[rows]), clear[rows],
                        mcfg.quantiles)
    return rows, art["meta"].predict(Z)


def compare_on_test_window(cfg, new_dir: Path, live_dir: Path) -> tuple[float, float, int] | None:
    """(new mean pinball, live mean pinball, n rows) on the rows of this run's test window
    that both models can predict. None if either model cannot be scored on this data."""
    from .clearsky import clearsky_feature
    from .data import prepare
    from .metrics import mean_pinball

    ds = prepare(cfg)
    df = ds.df
    _, _, te_mask = ds.split_masks()
    rows = np.where(te_mask & ds.score_mask())[0]
    clear = clearsky_feature(df)
    new = _predict_rows(new_dir, df, rows, clear, cfg)
    live = _predict_rows(live_dir, df, rows, clear, cfg)
    if new is None or live is None:
        return None
    common = np.intersect1d(new[0], live[0])
    if len(common) == 0:
        return None
    y = df[cfg.target].to_numpy("float32")[common]

    def on_common(rows_, preds):
        at = np.searchsorted(rows_, common)
        return {q: p[at] for q, p in preds.items()}

    return (mean_pinball(y, on_common(*new)), mean_pinball(y, on_common(*live)), len(common))


def _promotion_decision(cfg, repo: str, tag: str, token: str | None,
                        new_metrics: dict | None) -> tuple[bool, str]:
    skill = ((new_metrics or {}).get("test") or {}).get("skill_vs_neso_q50")
    if skill is None or not skill > 0:                # also catches NaN
        return False, f"test skill vs NESO is {skill}, not above 0"
    live = _fetch_live(repo, tag, token)
    if live is None:
        return True, f"no readable live model; test skill vs NESO {skill:.3f}"
    cmp = compare_on_test_window(cfg, Path(cfg.artifacts_dir), live)
    if cmp is None:
        return True, (f"live model cannot be scored on this data (target, quantiles or "
                      f"features changed); test skill vs NESO {skill:.3f}")
    new, old, n = cmp
    tol = float(os.getenv("GRIDSIGHT_PROMOTE_TOLERANCE", "0.02"))
    verdict = f"test mean_pinball new {new:.5f} vs live {old:.5f} on the same {n} rows"
    if new <= old * (1 + tol):
        return True, verdict
    return False, f"{verdict}, more than {tol:.0%} worse"


def push_model_if_better(cfg) -> bool:
    """Upload the freshly trained model in cfg.artifacts_dir to HF if it passes the
    promotion gate described in the module docstring."""
    repo = model_repo()
    art = Path(cfg.artifacts_dir)
    tag = art.name
    if not repo:
        logger.info("push_model: no model repo configured; skipping")
        return False
    new_metrics = json.loads((art / "metrics.json").read_text()) if (art / "metrics.json").exists() else None

    from huggingface_hub import HfApi
    token = _token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)

    # manual override for ad-hoc runs only; the scheduled retrain does not set it
    if os.getenv("GRIDSIGHT_FORCE_PROMOTE", "0").strip() in ("1", "true", "True"):
        ok, reason = True, "forced by GRIDSIGHT_FORCE_PROMOTE (gate bypassed)"
    else:
        ok, reason = _promotion_decision(cfg, repo, tag, token, new_metrics)
    if not ok:
        logger.warning(f"push_model[{tag}]: NOT promoted, {reason}; keeping the live model")
        return False

    logger.info(f"push_model[{tag}]: promoting, {reason}")
    api.upload_folder(
        folder_path=str(art), repo_id=repo, repo_type="model", path_in_repo=tag,
        allow_patterns=["stack.joblib", "tcn.pt", "metrics.json"],
        commit_message=f"promote {tag}: {reason}",
    )
    logger.success(f"push_model[{tag}]: uploaded -> hf://{repo}/{tag}")
    return True

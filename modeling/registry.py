"""HuggingFace model registry — pull the live model for serving, push a retrained one
only if it beats the current one.
Repo layout:   
1) model_12h/{stack.joblib, tcn.pt, metrics.json}
2)model_6h/ {stack.joblib, tcn.pt, metrics.json}

Repo id resolves from GRIDSIGHT_MODEL_HF_REPO, else derived from GRIDSIGHT_BRONZE_HF_REPO
(bronze -> model). Serving pulls the latest at load time (falls back to the baked-in
artifacts if HF is unset/unreachable). Retrain promotes on lower test mean_pinball
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from loguru import logger

_PROMOTE_KEY = ("test", "mean_pinball")     # lower is better

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

def _metric(metrics: dict | None) -> float:
    if not metrics:
        return float("inf")
    split, key = _PROMOTE_KEY
    try:
        return float(metrics[split][key])
    except (KeyError, TypeError, ValueError):
        return float("inf")

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

def push_model_if_better(cfg) -> bool:
    """Upload the freshly trained model in cfg.artifacts_dir to HF — but only if its
    test mean_pinball is lower than the current HF model's (or none exists yet)"""
    repo = model_repo()
    art = Path(cfg.artifacts_dir)
    tag = art.name
    if not repo:
        logger.info("push_model: no model repo configured; skipping")
        return False
    new_metrics = json.loads((art / "metrics.json").read_text()) if (art / "metrics.json").exists() else None
    new_score = _metric(new_metrics)

    from huggingface_hub import HfApi, hf_hub_download
    token = _token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)

    cur_metrics = None
    try:
        p = hf_hub_download(repo_id=repo, repo_type="model", filename=f"{tag}/metrics.json", token=token)
        cur_metrics = json.loads(Path(p).read_text())
    except Exception:
        cur_metrics = None                            
    cur_score = _metric(cur_metrics)

    split, key = _PROMOTE_KEY
    # Under a rolling val/test split the test window slides every run, so week-over-week
    # test scores aren't strictly comparable and the lower-is-better gate is unreliable.
    # GRIDSIGHT_FORCE_PROMOTE=1 (set by the weekly retrain) always ships the latest model
    # — the one trained on the most data. Leave it unset to keep the gate for ad-hoc runs.
    force = os.getenv("GRIDSIGHT_FORCE_PROMOTE", "0").strip() in ("1", "true", "True")
    if not force and not (new_score < cur_score):
        logger.warning(f"push_model[{tag}]: NOT promoted — new {split}.{key}={new_score:.5f} "
                       f">= current {cur_score:.5f}; keeping the live model")
        return False
    if force:
        logger.info(f"push_model[{tag}]: force-promote (gate bypassed) "
                    f"{split}.{key} {cur_score:.5f} -> {new_score:.5f}")

    logger.info(f"push_model[{tag}]: promoting — {split}.{key} {cur_score:.5f} -> {new_score:.5f}")
    api.upload_folder(
        folder_path=str(art), repo_id=repo, repo_type="model", path_in_repo=tag,
        allow_patterns=["stack.joblib", "tcn.pt", "metrics.json"],
        commit_message=f"promote {tag}: {split}.{key} {cur_score:.5f} -> {new_score:.5f}",
    )
    logger.success(f"push_model[{tag}]: uploaded -> hf://{repo}/{tag}")
    return True

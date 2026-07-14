"""Upload local Gold feature store -> HF dataset repo (gridsight-gold)"""
from __future__ import annotations
from loguru import logger
from .common import GOLD_LOCAL_DIR, GOLD_HF_REPO, GOLD_TABLE, hf_token

def upload_gold_to_hf(repo: str | None = None) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub required: pip install huggingface_hub")
        return

    repo = repo or GOLD_HF_REPO
    folder = GOLD_LOCAL_DIR
    if not folder.exists():
        logger.error(f"Local folder not found: {folder}")
        return

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    logger.info(f"Syncing {folder} -> hf://datasets/{repo}")
    # Mirror local onto the repo: upload the current gold_features_* tables and DELETE any
    # stale gold_features* files not present locally (e.g. the old single gold_features/
    # folder, or dropped partitions). One atomic commit; .gitattributes is untouched.
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="dataset",
                      delete_patterns=["gold_features*/**"],
                      ignore_patterns=["*.DS_Store", "**/.DS_Store"],
                      commit_message="gold sync: gold_features_12h + gold_features_6h (replace old gold_features)")
    logger.success(f"Synced {folder} -> hf://datasets/{repo}")

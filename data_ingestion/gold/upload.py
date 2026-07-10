"""Upload local Gold feature store -> your HF dataset repo (gridsight-gold)"""
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
    logger.info(f"Uploading {folder} -> hf://datasets/{repo}")
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="dataset",
                      commit_message="gold feature store upload")
    logger.success(f"Uploaded {folder} -> hf://datasets/{repo}")

"""Upload local Bronze dir -> HuggingFace dataset repo"""
from __future__ import annotations
from loguru import logger
from .common import BRONZE_LOCAL_DIR, BRONZE_HF_REPO, hf_token

def upload_to_hf(source: str | None = None) -> None:
    """upload local bronze dir to your HuggingFace dataset repo (BRONZE_HF_REPO)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub required: pip install huggingface_hub")
        return

    api = HfApi(token=hf_token)
    folder = BRONZE_LOCAL_DIR / source if source else BRONZE_LOCAL_DIR
    if not folder.exists():
        logger.error(f"Local folder not found: {folder}")
        return

    # make sure the dataset repo exists (no-op if it already does)
    api.create_repo(repo_id=BRONZE_HF_REPO, repo_type="dataset", exist_ok=True)

    logger.info(f"Uploading {folder} → hf://datasets/{BRONZE_HF_REPO}")
    path_in_repo = source if source else ""
    api.upload_folder(
        folder_path=str(folder),
        repo_id=BRONZE_HF_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        ignore_patterns=["_state.json"],
        commit_message=f"bronze upload: {source or 'all'}",
    )
    logger.success(f"Uploaded {folder} to HF")

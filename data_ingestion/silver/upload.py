from __future__ import annotations
from loguru import logger
from .common import SILVER_LOCAL_DIR, SILVER_HF_REPO, hf_token

TABLES = ["silver_pv_live", "silver_ocf_pv", "silver_met_office_nwp", "silver_neso"]

def upload_silver_to_hf(table: str | None = None, repo: str | None = None) -> None:
    """Upload one Silver table (or the whole silver/ dir) to the Silver HF repo"""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub required: pip install huggingface_hub")
        return

    repo = repo or SILVER_HF_REPO
    folder = SILVER_LOCAL_DIR / table if table else SILVER_LOCAL_DIR
    if not folder.exists():
        logger.error(f"Local folder not found: {folder}")
        return

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)

    path_in_repo = table if table else ""
    logger.info(f"Uploading {folder} → hf://datasets/{repo}/{path_in_repo}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=f"silver upload: {table or 'all'}",
    )
    logger.success(f"Uploaded {folder} → hf://datasets/{repo}")

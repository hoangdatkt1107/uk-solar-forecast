from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GRIDSIGHT_",
        extra="ignore",
        case_sensitive=False
    )
    # API
    hf_token: str
    # Paths
    data_dir: Path = Field(default=Path("./data"))

    bronze_hf_repo: str
    # optional; if unset, derived from bronze_hf_repo (bronze -> silver/gold)
    silver_hf_repo: str | None = None
    gold_hf_repo: str | None = None

settings = Settings()
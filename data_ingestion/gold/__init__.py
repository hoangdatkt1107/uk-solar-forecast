"""
Run as a module:  python -m data_ingestion.gold --horizon-steps 48
"""
from .merge import merge_silver
from .build import build_gold, run
from .upload import upload_gold_to_hf
from .cli import main

__all__ = ["merge_silver", "build_gold", "run", "upload_gold_to_hf", "main"]

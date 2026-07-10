"""Silver layer package.

Run as a module:  python -m data_ingestion.silver --source all
"""
from .pv_live import build_silver_pv_live
from .ocf_pv import build_silver_ocf_pv
from .met_office import build_silver_met_office_nwp
from .neso import build_silver_neso
from .upload import upload_silver_to_hf
from .cli import main

__all__ = [
    "build_silver_pv_live",
    "build_silver_ocf_pv",
    "build_silver_met_office_nwp",
    "build_silver_neso",
    "upload_silver_to_hf",
    "main",
]

"""
Run as a module:   python -m data_ingestion.bronze --source met_office_nwp 
"""
from .pv_live import ingest_pv_live
from .ocf_pv import ingest_ocf
from .met_office import ingest_met_office_nwp, mirror_met_office_nwp_raw
from .neso import ingest_neso
from .upload import upload_to_hf
from .cli import main

__all__ = [
    "ingest_pv_live",
    "ingest_ocf",
    "ingest_met_office_nwp",
    "mirror_met_office_nwp_raw",
    "ingest_neso",
    "upload_to_hf",
    "main",
]

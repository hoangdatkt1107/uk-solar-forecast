"""Probabilistic solar-forecasting stack: TCN-Q + LGBM-Q -> Linear-Q.
Run:  python -m modeling --fast"""

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
import lightgbm as _lgbm  

from .config import ModelConfig
from .train import run
from .predict import predict_gold
# NOTE: import evaluate lazily (`from modeling.evaluate import make_charts`) so that
# `python -m modeling.evaluate` doesn't double-import and warn

__all__ = ["ModelConfig", "run", "predict_gold"]
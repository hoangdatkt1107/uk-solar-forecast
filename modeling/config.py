from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
try:
    from gridsight.config import settings
    _DATA_DIR = settings.data_dir
except Exception:
    _DATA_DIR = Path("./data")

QUANTILES: tuple[float] = (0.10, 0.5, 0.90)

NON_FEATURE_COLS = {
    "timestamp_utc", "target_mw", "target_cf",
    "pv_flag", "nwp_flag", "neso_flag", "ocf_flag", "has_full_history",
}

@dataclass
class ModelConfig:
    gold_dir = _DATA_DIR / "gold" / "gold_features"
    artifacts_dir: Path = Path("artifacts/model")
    target: str = "target_cf"
    quantiles = QUANTILES
    horizon_step: int = 24 # must match the Gold build horizon (12h)

    # Rolling holdout by default: the val/test windows slide with the data every retrain
    # (test = the most recent `test_weeks`, val = the `val_weeks` before it, train = the
    # rest and grows each week). Pin a fixed benchmark by setting absolute dates via env
    # GRIDSIGHT_VAL_START / GRIDSIGHT_TEST_START. See modeling/data.py::split_masks.
    val_start: str | None = None
    test_start: str | None = None
    val_weeks: int = 8
    test_weeks: int = 8

    day_light_only: bool = True

    daylight_only: bool = True         # train/score only sun-up slots; night is predicted 0
    seed: int = 42

    # TCN-Q
    seq_len: int = 126
    tcn_channels: tuple[int,...] = (64, 64, 64, 64, 64, 64)
    tcn_kernel: int = 3
    tcn_dropout: float = 0.1
    tcn_lr: float = 1e-3
    tcn_epochs: int = 30
    tcn_batch:int = 256
    torch_threads: int = 1

    lgbm_params: dict = field(default_factory=lambda:{
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbose": -1,
    })

    n_folds: int = 5

    def __post_init__(self):
        # gold table + artifacts are per-horizon so 6h and 12h can coexist
        hours = self.horizon_step // 2
        self.gold_dir = _DATA_DIR / "gold" / f"gold_features_{hours}h"
        self.artifacts_dir = Path(f"artifacts/model_{hours}h")
        # env overrides: absolute dates pin the split; weeks tune the rolling window
        self.val_start = os.getenv("GRIDSIGHT_VAL_START") or self.val_start
        self.test_start = os.getenv("GRIDSIGHT_TEST_START") or self.test_start
        self.val_weeks = int(os.getenv("GRIDSIGHT_VAL_WEEKS", self.val_weeks))
        self.test_weeks = int(os.getenv("GRIDSIGHT_TEST_WEEKS", self.test_weeks))

    def quantile_names(self) -> list[str]:
        return [f"q{int(q*100)}" for q in self.quantiles]
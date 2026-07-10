from __future__ import annotations
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
    antifacts_dir: Path = Path("artifacts/model")
    target: str = "target_cf"
    quantiles = QUANTILES
    horizon_step: int = 48 # must match the Gold build horizon

    val_start: str = "2024-07-01"
    test_start: str = "2024-10-01"

    day_light_only: bool = True

    daylight_only: bool = True         # train/score only sun-up slots; night is predicted 0
    seed: int = 42

    # ----- TCN-Q
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

    def quantile_names(self) -> list[str]:
        return [f"q{int(q*100)}" for q in self.quantiles]
    
print(Path(ModelConfig.gold_dir).resolve())
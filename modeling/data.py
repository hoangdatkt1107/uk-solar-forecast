from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

from .config import ModelConfig, NON_FEATURE_COLS

@dataclass
class Dataset:
    df: pd.DataFrame
    cfg: ModelConfig
    feature_columns: list[str]

    def score_mask(self) -> np.ndarray:
        """for evaluation because just daytime have solar enery"""
        if self.cfg.day_light_only and "is_daylight" in self.df.columns:
            return (self.df["is_daylight"] == 1).to_numpy()
        else:
            return np.ones(len(self.df), dtype=bool)

    def split_masks(self):
        """
        return boolean masks (train_data - val_data - test_data) for train/val/test splits expluding real data 
        """
        if "timestamp_utc" in self.df.columns:
            ts = self.df["timestamp_utc"]
            cfg = self.cfg
            if cfg.val_start and cfg.test_start:            # pinned absolute split
                val_start = pd.Timestamp(cfg.val_start, tz="UTC")
                test_start = pd.Timestamp(cfg.test_start, tz="UTC")
            else:                                           # rolling: slide with the data
                max_ts = pd.Timestamp(ts.max())
                if max_ts.tzinfo is None:
                    max_ts = max_ts.tz_localize("UTC")
                test_start = max_ts - pd.Timedelta(weeks=cfg.test_weeks)
                val_start = test_start - pd.Timedelta(weeks=cfg.val_weeks)
                logger.info(f"rolling split: train<{val_start.date()} | "
                            f"val {val_start.date()}->{test_start.date()} | "
                            f"test>={test_start.date()} (data max {max_ts.date()})")
            train_data = (ts < val_start).to_numpy()
            val_data = ((ts >= val_start) & (ts < test_start)).to_numpy()
            test_data = (ts >= test_start).to_numpy()
            return train_data, val_data, test_data
        
        else:
            logger.error("please make sure timestamp_utc exsist")
            return None
        
    
def load_gold(cfg: ModelConfig) -> pd.DataFrame:
    path_list = sorted(Path(cfg.gold_dir).rglob("*.parquet"))
    if not path_list:
        raise FileNotFoundError(
            f"No gold file under {cfg.gold_dir} /n"
            f"Built it first python -m data_ingestion.gold --horizon-steps {cfg.horizon_step}")
   
    df = pd.concat([pd.read_parquet(i) for i in path_list], ignore_index=True)
    df = df.sort_values("timestamp_utc", ascending=True).reset_index(drop=True)
    logger.info("Gold data is loaded")
    return df

def feature_columns(df: pd.DataFrame) -> list:
    """extract only useful feature for training"""
    features = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    return features

def prepare(cfg: ModelConfig) -> Dataset:
    df = load_gold(cfg)
    if cfg.target not in df.columns:
        raise KeyError(f"Target {cfg.target} not in Gold")
    df = df[df["has_full_history"] == 1]
    df = df[df[cfg.target].notna()].reset_index(drop=True) 
    return Dataset(df=df, feature_columns=feature_columns(df), cfg=cfg)

def make_sequences(value: np.ndarray, seq_len: int):
    n = value.shape[0]
    if n <= seq_len:
        return np.empty((0,seq_len, value.shape[1]), dtype="float32"), np.empty((0,), dtype="int64")
    window = np.lib.stride_tricks.sliding_window_view(value, seq_len, axis=0)
    seqs = np.ascontiguousarray(window.transpose(0,2,1)).astype("float32")
    end_idx = np.arange(seq_len-1, value.shape[0])
    return seqs, end_idx

class Standardizer:
    def fit(self, X: np.ndarray):
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[~np.isfinite(self.std_) | (self.std_ == 0)] = 1.0
        self.mean_ = np.nan_to_num(self.mean_)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isnan(X), self.mean_, X)
        return ((X - self.mean_) / self.std_).astype("float32")

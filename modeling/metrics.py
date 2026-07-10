from __future__ import annotations
import numpy as np  
import pdb


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    residual = y_true - y_pred
    result = np.maximum(q * residual, (q -1) * residual)
    return float(np.mean(result))

def mean_pinball(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    mean_pinball_list = []
    for q, p in preds.items():
        loss = float(np.mean(pinball_loss(y_true, p, q)))
        mean_pinball_list.append(loss)
    return float(np.mean(mean_pinball_list))

def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """return the fraction of rows where the true value is between the lower and upper quantiles"""
    is_covered = ((y_true >= lo) & (y_true <= hi))
    return float(np.mean(is_covered))

def crossing_rate(preds: dict[float, np.ndarray]) -> float:
    """return the fraction of rows where the quantiles cross each other 
        (should be 0)"""
    qs = sorted(preds)
    stack =np.stack([preds[i] for i in qs ], axis= 1)
    diff = np.diff(stack, axis=1)
    return float(np.mean(np.any(diff < -1e-9, axis=1)))

def skill_vs_baseline(y_true, q50_pred, baseline_pred) -> float:
    """MAE skill score of the q50 forecast vs a point baseline (e.g. NESO). >0 = better."""
    mae_model = np.mean(np.abs(y_true - q50_pred))
    mae_base = np.mean(np.abs(y_true - baseline_pred))
    return float(1.0 - mae_model / mae_base) if mae_base > 0 else float("nan")

def report(y_true: np.ndarray, preds: dict[float, np.ndarray],
           quantiles=(0.1, 0.5, 0.9)) -> dict:
    lo, hi = quantiles[0], quantiles[-1]
    out = {f"pinball_q{int(q*100)}": pinball_loss(y_true, preds[q], q) for q in quantiles}
    out["mean_pinball"] = mean_pinball(y_true, preds)
    out["coverage_%d-%d" % (int(lo*100), int(hi*100))] = coverage(y_true, preds[lo], preds[hi])
    out["target_coverage"] = hi - lo
    out["crossing_rate"] = crossing_rate(preds)
    return out

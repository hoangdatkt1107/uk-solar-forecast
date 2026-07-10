from __future__ import annotations
import numpy as np

def assemble_meta_X(tcn: dict[float, np.ndarray], lgbm: dict[float, np.ndarray], 
                    clearsky: np.ndarray, quantile: tuple) -> np.ndarray:
    col =  [tcn[q] for q in quantile] + [ lgbm[q] for q in quantile] + [clearsky]
    return np.column_stack(col)

class LinearQuantileStacker:
    def __init__(self, quantiles = (0.1, 0.5, 0.9)):
        self.quantiles = quantiles
        self.model_ : dict[float, object] = {}
    
    def fit(self, Z: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import QuantileRegressor
        for q in self.quantiles:
            self.model_[q] = QuantileRegressor(quantile=q, alpha=0.0,
                                               solver="highs").fit(Z,y)
        return self
    
    def predict(self, Z: np.ndarray) -> dict[float, np.ndarray]:
        P = np.column_stack([self.model_[q].predict(Z) for q in self.quantiles])
        P = np.sort(P, axis=1)
        result = {q: P[:,i] for i, q in enumerate(self.quantiles)}
        return result

            


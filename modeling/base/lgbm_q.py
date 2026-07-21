from __future__ import annotations
import numpy as np

class LGBMQuantile:
    def __init__(self, quantiles=(0.1, 0.5, 0.9), params: dict | None = None, seed: int = 42):
        self.quantiles = tuple(quantiles)
        self.params = dict(params or {})
        self.seed = seed
        self.models_: dict[float, object] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: tuple | None = None) -> "LGBMQuantile":
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        for q in self.quantiles:
            model =LGBMRegressor(objective='quantile', alpha=q,
                                  random_state=self.seed, **self.params)
            callback, early_stop = [], eval_set
            if early_stop is not None:
                callback = [early_stopping(50, verbose=True), log_evaluation(0)]
            model.fit(X, y, eval_set=[early_stop] if early_stop is not None else None, callbacks=callback or None)
            self.models_[q] = model
        return self
    
    def predict(self, X: np.ndarray) -> dict[float, np.ndarray]:
        prediction: dict = {}
        for q, m in self.models_.items():
            prediction[q] = m.predict(X).astype("float32")
        return prediction
    
    def feature_importance(self):
        result: dict = {}
        for q,m in self.models_.items():
            result[q] = m.feature_importances_
        return result

        
import numpy as np
import torch
from typing import Optional, Dict, Any
from xgboost import XGBRegressor
from sbtab.models.field.neural.time_embedding import FourierTime

class XGBField:
    """
    The drift field is based on XGBoost. 
    Trains D independent models (one for each dimension).
    """
    def __init__(
        self, 
        dim: int, 
        time_features: int = 16, 
        max_freq: float = 20.0,
        xgb_params: Optional[Dict[str, Any]] = None
    ):
        self.dim = dim
        self.time_features = time_features
        self.time_embedder = FourierTime(features=time_features, max_freq=max_freq)
        
        self.params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "verbosity": 0,
            "random_state": 42,
            "n_jobs": -1
        }
        if xgb_params:
            self.params.update(xgb_params)
            
        self.models = [XGBRegressor(**self.params) for _ in range(self.dim)]

    def _prepare_features(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        t_tensor = torch.from_numpy(t.astype(np.float32))
        with torch.no_grad():
            t_emb = self.time_embedder(t_tensor).numpy()
        return np.concatenate([x, t_emb], axis=1)

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray):
        features = self._prepare_features(x, t)
        for i in range(self.dim):
            self.models[i].fit(features, y[:, i])
        return self

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        features = self._prepare_features(x, t)
        preds = []
        for i in range(self.dim):
            preds.append(self.models[i].predict(features))
        return np.stack(preds, axis=1).astype(np.float32)
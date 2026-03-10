import numpy as np
import torch
from typing import Optional, Dict, Any
from catboost import CatBoostRegressor, Pool
from sbtab.models.field.neural.time_embedding import FourierTime

class CatBoostField:
    """
    A CatBoost-based drift field using MultiRMSE.
    One model for all dimensions.
    """
    def __init__(
        self, 
        dim: int, 
        time_features: int = 16, 
        max_freq: float = 20.0,
        cat_params: Optional[Dict[str, Any]] = None
    ):
        self.dim = dim
        self.time_features = time_features
        self.time_embedder = FourierTime(features=time_features, max_freq=max_freq)
        
        self.params = {
            "loss_function": "MultiRMSE",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "random_state": 42,
            "task_type": "CPU",
            "verbose": 100
        }
        if cat_params:
            self.params.update(cat_params)
            
        self.model = CatBoostRegressor(**self.params)

    def _prepare_features(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        t_tensor = torch.from_numpy(t.astype(np.float32))
        with torch.no_grad():
            t_emb = self.time_embedder(t_tensor).numpy()
        return np.concatenate([x, t_emb], axis=1).astype(np.float32)

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray):
        features = self._prepare_features(x, t)
        train_pool = Pool(data=features, label=y)
        self.model.fit(train_pool)
        return self

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        features = self._prepare_features(x, t)
        preds = self.model.predict(features)
        return np.asarray(preds, dtype=np.float32)
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

from sbtab.models.field.neural.time_embedding import FourierTime


@dataclass
class CatBoostContinuousFieldConfig:
    """
    CatBoost field approximator for continuous time.
    Predicts a vector drift in R^dim using MultiRMSE.
    """
    iterations: int = 2000
    depth: int = 8
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0

    task_type: Literal["CPU", "GPU"] = "CPU"
    thread_count: int = -1
    random_seed: int = 0
    verbose: bool = False
    allow_writing_files: bool = False

    # Continuous time specific parameters
    time_features: int = 16
    max_freq: float = 20.0
    feature_mode: Literal["x_t", "x_x0_t"] = "x_t"


class CatBoostContinuousField:
    """
    A single CatBoost regressor conditioned on continuous time t.
    Uses Fourier features to embed t.
    """

    def __init__(self, dim: int, cfg: CatBoostContinuousFieldConfig):
        self.dim = int(dim)
        self.cfg = cfg
        self.time_embedder = FourierTime(features=cfg.time_features, max_freq=cfg.max_freq)
        self.model = None
        self._checked_deps = False

    def _check_deps(self):
        if self._checked_deps:
            return
        try:
            import catboost  # noqa
        except ImportError as e:
            raise ImportError(
                "CatBoostContinuousField requires catboost.\n"
                "Install with: pip install catboost"
            ) from e
        self._checked_deps = True

    def _build_features(
        self,
        x: np.ndarray,
        t: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies Fourier time embedding and concatenates with input features.
        If feature_mode is 'x_x0_t', it also concatenates the context x0.
        """
        t_tensor = torch.from_numpy(np.asarray(t, dtype=np.float32).reshape(-1))
        with torch.no_grad():
            t_emb = self.time_embedder(t_tensor).numpy()

        parts =[x, t_emb]

        if self.cfg.feature_mode == "x_x0_t":
            if x0 is None:
                raise ValueError("feature_mode 'x_x0_t' requires x0 to be provided")
            parts.append(x0)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, X_feat_raw: np.ndarray, t: np.ndarray, y: np.ndarray, x0: Optional[np.ndarray] = None):
        """Fits the single multi-output CatBoost model."""
        self._check_deps()
        from catboost import CatBoostRegressor, Pool

        X_feat = self._build_features(X_feat_raw, t, x0=x0)
        
        boosting_type = "Plain" if self.cfg.task_type == "GPU" else "Ordered"
        
        self.model = CatBoostRegressor(
            iterations=self.cfg.iterations,
            depth=self.cfg.depth,
            learning_rate=self.cfg.learning_rate,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            loss_function="MultiRMSE" if self.dim > 1 else "RMSE",
            task_type=self.cfg.task_type,
            boosting_type=boosting_type,
            thread_count=self.cfg.thread_count,
            random_seed=self.cfg.random_seed,
            verbose=self.cfg.verbose,
            allow_writing_files=self.cfg.allow_writing_files,
        )

        train_pool = Pool(data=X_feat, label=y)
        self.model.fit(train_pool)
        return self

    def predict(self, x: np.ndarray, t: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicts the multi-dimensional drift vector."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        
        X_feat = self._build_features(x, t, x0=x0)
        pred = self.model.predict(X_feat)
        
        if self.dim == 1 and pred.ndim == 1:
            pred = pred.reshape(-1, 1)
            
        return np.asarray(pred, dtype=np.float32)
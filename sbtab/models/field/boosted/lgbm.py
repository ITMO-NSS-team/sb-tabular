from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LGBMStepConfig:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    n_jobs: int = -1
    random_state: int = 42
    extra_params: Dict[str, Any] = field(default_factory=dict)


class LGBMStepModel:

    def __init__(self, dim: int, cfg: LGBMStepConfig):
        self.dim = int(dim)
        self.cfg = cfg
        self._models: Optional[List[Any]] = None

    def _make_one(self):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("LGBMStepModel requires lightgbm. Install: pip install lightgbm") from e

        params = dict(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            num_leaves=self.cfg.num_leaves,
            min_child_samples=self.cfg.min_child_samples,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            verbose=-1,
            **self.cfg.extra_params,
        )
        return lgb.LGBMRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMStepModel":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        assert X.ndim == 2 and y.ndim == 2
        assert X.shape[0] == y.shape[0]
        assert y.shape[1] == self.dim

        self._models = []
        for d in range(self.dim):
            m = self._make_one()
            m.fit(X, y[:, d])
            self._models.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._models is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float32)
        preds = np.stack([m.predict(X) for m in self._models], axis=1)
        return preds.astype(np.float32)

    def is_fitted(self) -> bool:
        return self._models is not None

    def clone(self) -> "LGBMStepModel":
        return LGBMStepModel(dim=self.dim, cfg=deepcopy(self.cfg))

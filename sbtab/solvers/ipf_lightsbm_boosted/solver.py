from __future__ import annotations

import math
from typing import List, Optional, Any

import numpy as np
import pandas as pd

from sbtab.models.field.boosted.lgbm import LGBMStepConfig, LGBMStepModel
from sbtab.solvers.ipf_lightsbm_boosted.config import LightSBMBoostedConfig


class _LGBMDriftField:

    def __init__(self, dim: int, cfg: LightSBMBoostedConfig):
        self.dim = dim
        lgbm_cfg = LGBMStepConfig(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            n_jobs=cfg.n_jobs,
            random_state=cfg.seed,
            extra_params=cfg.lgbm_extra,
        )
        self._model = LGBMStepModel(dim=dim, cfg=lgbm_cfg)

    def fit(self, x_t: np.ndarray, t: np.ndarray, target: np.ndarray) -> "_LGBMDriftField":
        t_col = t.reshape(-1, 1).astype(np.float32)
        X = np.concatenate([x_t, t_col], axis=1)
        self._model.fit(X, target)
        return self

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        t_col = t.reshape(-1, 1).astype(np.float32)
        X = np.concatenate([x.astype(np.float32), t_col], axis=1)
        return self._model.predict(X)

    def is_fitted(self) -> bool:
        return self._model.is_fitted()


class LightSBMBoostedSolver:

    def __init__(self, dim: int, cfg: LightSBMBoostedConfig):
        self.dim = int(dim)
        self.cfg = cfg
        self._field: Optional[_LGBMDriftField] = None
        self._fitted = False

    def _to_numpy(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(dtype=np.float32, copy=True)
        return np.asarray(data, dtype=np.float32)

    def _build_training_set(
        self, x_data: np.ndarray, rng: np.random.Generator
    ):
        cfg = self.cfg
        N_data = x_data.shape[0]
        N = cfg.n_pairs

        idx = rng.integers(0, N_data, size=N)
        x_1 = x_data[idx]

        x_0 = rng.standard_normal((N, self.dim)).astype(np.float32)

        t = (rng.random(N) * (1.0 - cfg.safe_t) + cfg.safe_t * 0.5).astype(np.float32)

        noise = rng.standard_normal((N, self.dim)).astype(np.float32)
        t_col = t[:, None]
        x_t = (
            x_1 * t_col
            + x_0 * (1.0 - t_col)
            + np.sqrt(cfg.epsilon * t_col * (1.0 - t_col)) * noise
        )

        target = (x_1 - x_t) / (1.0 - t_col)

        return x_t, t, target

    def fit(self, train: pd.DataFrame | np.ndarray) -> "LightSBMBoostedSolver":
        x_data = self._to_numpy(train)
        if x_data.ndim != 2 or x_data.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {x_data.shape}")

        rng = np.random.default_rng(self.cfg.seed)

        x_t, t, target = self._build_training_set(x_data, rng)

        self._field = _LGBMDriftField(dim=self.dim, cfg=self.cfg)
        self._field.fit(x_t, t, target)

        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        if not self._fitted or self._field is None:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive.")

        rng = np.random.default_rng(seed if seed is not None else self.cfg.seed + 1)

        x = rng.standard_normal((n, self.dim)).astype(np.float32)

        K = self.cfg.n_euler_steps
        dt = 1.0 / K
        eps = self.cfg.epsilon

        for i in range(K):
            t_val = float(i) / float(K)
            t_arr = np.full(n, t_val, dtype=np.float32)

            drift = self._field.predict(x, t_arr)

            noise = rng.standard_normal((n, self.dim)).astype(np.float32)
            x = x + drift * dt + math.sqrt(eps * dt) * noise

        return x

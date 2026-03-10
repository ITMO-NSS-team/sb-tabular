from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from sbtab.models.field.boosted.catboost_discrete_field import (
    CatBoostDiscreteFieldConfig,
    CatBoostTimeDiscretizedField,
)


@dataclass
class IMFDSBMBoostConfig:

    fb_sequence: Sequence[Literal["b", "f"]] = ("b", "f", "b", "f", "b")

    num_steps: int = 32
    sigma: float = 0.10
    eps: float = 1e-3

    first_coupling: Literal["ref", "ind"] = "ind"

    n_noise_per_pair: int = 1

    noise: bool = True

    seed: int = 0

    catboost: CatBoostDiscreteFieldConfig = CatBoostDiscreteFieldConfig()

    sample_direction: Literal["f", "b"] = "f"


class IMFDSBMBoostSolver:

    def __init__(self, dim: int, cfg: IMFDSBMBoostConfig):

        self.dim = int(dim)
        self.cfg = cfg

        self.columns_: Optional[list[str]] = None

        self.t_grid = self._make_t_grid(cfg.num_steps, cfg.eps)

        self.field_f: Optional[CatBoostTimeDiscretizedField] = None
        self.field_b: Optional[CatBoostTimeDiscretizedField] = None

        self._rng = np.random.default_rng(cfg.seed)

        self._x_data: Optional[np.ndarray] = None
        self._x_ref: Optional[np.ndarray] = None

    # -------------------------------------------------------

    @staticmethod
    def _make_t_grid(N: int, eps: float):

        t = (np.arange(N, dtype=np.float32) + 0.5) / float(N)

        t = np.clip(t, eps, 1 - eps)

        return t

    # -------------------------------------------------------

    def _dsbm_train_tuple(self, z0, z1, t, fb):

        sigma = self.cfg.sigma

        noise = self._rng.normal(size=z0.shape).astype(np.float32)

        xt = t * z1 + (1 - t) * z0

        xt += sigma * np.sqrt(t * (1 - t)) * noise

        delta = z1 - z0

        if fb == "f":
            target = delta - sigma * np.sqrt(t / (1 - t)) * noise
        else:
            target = -delta - sigma * np.sqrt((1 - t) / t) * noise

        return xt.astype(np.float32), target.astype(np.float32)

    # -------------------------------------------------------

    def _train_direction_field(self, fb, z0, z1):

        field = CatBoostTimeDiscretizedField(
            dim=self.dim,
            t_grid=self.t_grid,
            cfg=self.cfg.catboost,
        )

        for k, t in enumerate(self.t_grid):

            xt, target = self._dsbm_train_tuple(z0, z1, float(t), fb)

            field.fit_step(k, xt, target, x0=z0)

        return field

    # -------------------------------------------------------

    def _sample_with_direction(self, zstart, direction):

        field = self.field_f if direction == "f" else self.field_b

        N = self.cfg.num_steps
        dt = 1.0 / N
        sigma = self.cfg.sigma

        x = zstart.astype(np.float32)

        x0 = x.copy()

        step_indices = range(N) if direction == "f" else range(N - 1, -1, -1)

        for k in step_indices:

            drift = field.predict_step(k, x, x0=x0)

            x = x + drift * dt

            if self.cfg.noise:
                x += sigma * np.sqrt(dt) * self._rng.normal(size=x.shape)

        return x

    # -------------------------------------------------------

    def fit(self, train_df):

        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("fit expects DataFrame")

        self.columns_ = list(train_df.columns)

        x_data = train_df.to_numpy(dtype=np.float32)

        x_ref = self._rng.normal(size=x_data.shape).astype(np.float32)

        self._x_data = x_data
        self._x_ref = x_ref

        prev_fb = None

        for it, fb in enumerate(self.cfg.fb_sequence):

            if it == 0:

                if self.cfg.first_coupling == "ref":

                    z0 = x_ref
                    z1 = z0 + self._rng.normal(size=z0.shape) * self.cfg.sigma

                else:

                    perm = self._rng.permutation(len(x_data))
                    z0 = x_ref
                    z1 = x_data[perm]

            else:

                zstart = x_ref if prev_fb == "f" else x_data

                zend = self._sample_with_direction(zstart, prev_fb)

                if prev_fb == "f":
                    z0, z1 = zstart, zend
                else:
                    z0, z1 = zend, zstart

            field = self._train_direction_field(fb, z0, z1)

            if fb == "f":
                self.field_f = field
            else:
                self.field_b = field

            prev_fb = fb

        return self

    # -------------------------------------------------------

    def sample(self, n, seed=None):

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        z0 = self._rng.normal(size=(n, self.dim)).astype(np.float32)

        x = self._sample_with_direction(z0, self.cfg.sample_direction)

        return x

    def sample_df(self, n, seed=None):

        x = self.sample(n, seed)

        return pd.DataFrame(x, columns=self.columns_)
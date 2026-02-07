from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from sbtab.baselines.base import BaselineGenerativeModel, BaselineFitInfo, ArrayLike


TaskType = Literal["auto", "classification", "regression"]


@dataclass
class TabPFGenConfig:
    """
    Wrapper config around sebhaan/TabPFGen.

    TabPFGen constructor parameters (from the repo README):
      - n_sgld_steps
      - sgld_step_size
      - sgld_noise_scale
      - device
    :contentReference[oaicite:2]{index=2}
    """
    target_col: str
    task: TaskType = "auto"

    # TabPFGen core sampling controls
    n_sgld_steps: int = 1000
    sgld_step_size: float = 0.01
    sgld_noise_scale: float = 0.01
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # Task-specific generation options (from README examples)
    balance_classes: bool = True      # classification
    use_quantiles: bool = True        # regression :contentReference[oaicite:3]{index=3}

    seed: int = 0


class TabPFGenGenerative(BaselineGenerativeModel):
    """
    Baseline generative model using sebhaan/TabPFGen.

    fit(df):
      - requires a DataFrame so we can separate X and y by target_col
      - stores train X,y
      - instantiates TabPFGen generator (no extra training)

    sample(n):
      - calls:
          generate_classification(X, y, n_samples=n, balance_classes=...)
        or
          generate_regression(X, y, n_samples=n, use_quantiles=...)
      - returns a DataFrame with the same columns as input
    """

    def __init__(self, cfg: TabPFGenConfig):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg

        self._fitted_df: bool = False
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._feature_cols: Optional[list[str]] = None
        self._task: Optional[Literal["classification", "regression"]] = None

        self._generator: Any = None

    def fit(self, data: ArrayLike, **kwargs: Any) -> "TabPFGenGenerative":
        if not isinstance(data, pd.DataFrame):
            raise ValueError("TabPFGenGenerative requires a pandas DataFrame input (to locate target_col).")

        df = data.copy()
        cols = list(df.columns)
        self.columns_ = cols
        self._fitted_df = True

        if self.cfg.target_col not in df.columns:
            raise ValueError(f"target_col='{self.cfg.target_col}' not found in columns: {cols}")

        feature_cols = [c for c in cols if c != self.cfg.target_col]
        if len(feature_cols) < 1:
            raise ValueError("No feature columns after removing target_col.")
        self._feature_cols = feature_cols

        # Extract X, y
        X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        y = df[self.cfg.target_col].to_numpy(copy=True)

        # Decide task
        task = self.cfg.task
        if task == "auto":
            s = df[self.cfg.target_col]
            # Heuristic:
            # - non-numeric => classification
            # - bool => classification
            # - small integer set => classification
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
                task = "classification"
            else:
                s_num = pd.to_numeric(s, errors="coerce")
                nunique = int(pd.Series(s_num).nunique(dropna=True))
                if pd.api.types.is_integer_dtype(s_num) and nunique <= 20:
                    task = "classification"
                else:
                    task = "regression"

        if task not in ("classification", "regression"):
            raise ValueError(f"Unsupported task after resolution: {task}")
        self._task = task

        # For regression ensure numeric y
        if self._task == "regression":
            y = pd.to_numeric(df[self.cfg.target_col], errors="coerce").to_numpy(dtype=np.float32, copy=True)
            if np.isnan(y).any():
                raise ValueError("Regression target contains NaNs after numeric conversion.")

        self._X_train = X
        self._y_train = y

        self.fit_info_ = BaselineFitInfo(
            n_rows=int(df.shape[0]),
            n_cols=int(df.shape[1]),
            columns=cols,
        )

        # Instantiate TabPFGen generator
        try:
            from tabpfgen import TabPFGen  # pip install tabpfgen :contentReference[oaicite:4]{index=4}
        except Exception as e:
            raise ImportError("TabPFGenGenerative requires `tabpfgen`. Install: pip install tabpfgen") from e

        self._generator = TabPFGen(
            n_sgld_steps=int(self.cfg.n_sgld_steps),
            sgld_step_size=float(self.cfg.sgld_step_size),
            sgld_noise_scale=float(self.cfg.sgld_noise_scale),
            device=str(self.cfg.device),
        )
        return self

    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> ArrayLike:
        if self.fit_info_ is None or self._X_train is None or self._y_train is None or self._feature_cols is None or self._task is None:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive")

        # TabPFGen does not expose an explicit 'seed' parameter in the README API,
        # so we seed numpy for any stochasticity on our side.
        if seed is not None:
            np.random.seed(int(seed))

        if self._task == "classification":
            Xs, ys = self._generator.generate_classification(
                self._X_train,
                self._y_train,
                n_samples=int(n),
                balance_classes=bool(self.cfg.balance_classes),
            )
        else:
            Xs, ys = self._generator.generate_regression(
                self._X_train,
                self._y_train,
                n_samples=int(n),
                use_quantiles=bool(self.cfg.use_quantiles),
            )

        Xs = np.asarray(Xs)
        ys = np.asarray(ys)

        # Build output DataFrame in original column order
        out = pd.DataFrame(columns=self.columns_)
        for j, c in enumerate(self._feature_cols):
            out[c] = Xs[:, j]
        out[self.cfg.target_col] = ys
        out = out[self.columns_]
        return out

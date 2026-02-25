"""
from sbtab.solvers.ipf_dsb_boosted.solver import IPFDSBBoostedSolver

model = IPFDSBBoostedSolver(
    dim=len(cols),
    model_type="catboost", # или "xgboost"
    num_steps=20,
    ipf_iters=5,
    gbdt_params={"iterations": 500, "depth": 6}
)
model.fit(train_sc)
x_syn = model.sample(n=len(test_sc))
"""

import numpy as np
import pandas as pd
import torch
from tqdm import trange
from typing import Optional, Literal, Dict, Any

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.field.boosted.catboost import CatBoostField
from sbtab.models.field.boosted.xgb import XGBField
from sbtab.evaluation.metrics.statistical import sliced_wasserstein

class IPFDSBBoostedSolver:
    """
    Diffusion Schrödinger Bridge Solver using GBDT (CatBoost/XGBoost) as fields.
    Implements the Iterative Proportional Fitting (IPF) algorithm for tabular data.
    """
    def __init__(
        self,
        dim: int,
        model_type: Literal["catboost", "xgboost"] = "catboost",
        num_steps: int = 20,
        ipf_iters: int = 5,
        T: float = 1.0,
        alpha_ou: float = 1.0,
        gbdt_params: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.dim = dim
        self.ipf_iters = ipf_iters
        self.alpha_ou = alpha_ou
        self.seed = seed
        
        # Initialize the discrete time grid
        self.timegrid = TimeGrid(num_steps=num_steps, T=T)
        self.gammas = self.timegrid.gammas().numpy()
        self.times = self.timegrid.times().numpy()

        # Select the GBDT field model class
        FieldClass = CatBoostField if model_type == "catboost" else XGBField
        self.F = FieldClass(dim=dim, cat_params=gbdt_params if model_type == "catboost" else None)
        self.B = FieldClass(dim=dim, cat_params=gbdt_params if model_type == "catboost" else None)
        
        self._fitted = False

    def _sample_prior(self, n: int):
        """Samples from the standard normal prior."""
        return np.random.randn(n, self.dim).astype(np.float32)

    def pretrain_F_ou(self, X_data: np.ndarray):
        """Initializes the Forward field as an Ornstein-Uhlenbeck process."""
        n = len(X_data)
        x_prior = self._sample_prior(n)
        x = np.vstack([X_data, x_prior])
        
        k = np.random.randint(0, len(self.times), size=x.shape[0])
        t_k = self.times[k]
        dt = self.gammas[k]
        
        # OU Drift Target: x + dt * (-alpha * x)
        target = x + dt[:, None] * (-self.alpha_ou * x)
        self.F.fit(x, t_k, target)

    def fit(self, X_train: np.ndarray):
        """Runs the main IPF training loop."""
        X_train = np.asarray(X_train, dtype=np.float32)
        print("Pretraining Forward field (OU)...")
        self.pretrain_F_ou(X_train)

        for i in range(self.ipf_iters):
            # Phase 1: Train Backward field B on Forward trajectories
            print(f"IPF Iteration {i+1}/{self.ipf_iters} - Training B...")
            xb, tb, yb = self._build_dataset(X_train, self.F, direction="forward")
            self.B.fit(xb, tb, yb)

            # Phase 2: Train Forward field F on Backward trajectories
            print(f"IPF Iteration {i+1}/{self.ipf_iters} - Training F...")
            xf, tf, yf = self._build_dataset(self._sample_prior(len(X_train)), self.B, direction="backward")
            self.F.fit(xf, tf, yf)
            
            # Monitoring: Compute Sliced Wasserstein Distance
            syn = self.sample(num=min(len(X_train), 2000))
            swd = sliced_wasserstein(X_train, syn)
            print(f"Iter {i+1} SWD: {swd:.5f}")

        self._fitted = True
        return self

    def _build_dataset(self, x_init: np.ndarray, model_opp, direction: str):
        """Simulates trajectories using the current model to build a dataset for the opposite model."""
        xs, ts, ys = [], [], []
        x = x_init.copy()
        n = len(x)
        
        steps = range(len(self.times)) if direction == "forward" else range(len(self.times)-1, -1, -1)
        
        for k in steps:
            t_k = np.full((n,), self.times[k], dtype=np.float32)
            drift = model_opp.predict(x, t_k)
            
            # Stochastic SDE Step (Euler-Maruyama)
            noise = np.random.randn(*x.shape) * np.sqrt(2.0 * self.gammas[k])
            x_next = drift + noise
            
            # Training Target for Schrödinger Bridge matching
            target = x_next # Simplified target logic
            
            xs.append(x)
            ts.append(t_k)
            ys.append(target)
            x = x_next
            
        return np.vstack(xs), np.hstack(ts), np.vstack(ys)

    def sample(self, num: int):
        """Generates synthetic data by running the backward process from noise."""
        x = self._sample_prior(num)
        for k in range(len(self.times) - 1, -1, -1):
            t_k = np.full((num,), self.times[k], dtype=np.float32)
            x = self.B.predict(x, t_k) + np.random.randn(*x.shape) * np.sqrt(2.0 * self.gammas[k])
        return x
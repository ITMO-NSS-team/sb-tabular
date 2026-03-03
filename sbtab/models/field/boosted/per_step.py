from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch


class PerStepField:

    def __init__(self, num_steps: int, step_model_factory: Callable[[], object]):
        self.num_steps = int(num_steps)
        self._models: List[object] = [step_model_factory() for _ in range(self.num_steps)]

    def fit(self, k: int, X: np.ndarray, y: np.ndarray) -> "PerStepField":
        self._models[k].fit(X, y)
        return self

    def predict(self, k: int, X: np.ndarray) -> np.ndarray:
        return self._models[k].predict(X)

    def predict_tensor(
        self,
        x: torch.Tensor,
        k: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        y_np = self.predict(k, x_np)
        out = torch.from_numpy(y_np)
        target_device = device if device is not None else x.device
        return out.to(target_device, dtype=torch.float32)

    def is_fitted(self, k: int) -> bool:
        m = self._models[k]
        if hasattr(m, "is_fitted"):
            return m.is_fitted()
        return True

    def all_fitted(self) -> bool:
        return all(self.is_fitted(k) for k in range(self.num_steps))

    def get_model(self, k: int) -> object:
        return self._models[k]

    @classmethod
    def from_lgbm(
        cls,
        num_steps: int,
        dim: int,
        cfg: object,
    ) -> "PerStepField":
        from sbtab.models.field.boosted.lgbm import LGBMStepModel
        return cls(num_steps=num_steps, step_model_factory=lambda: LGBMStepModel(dim=dim, cfg=cfg))

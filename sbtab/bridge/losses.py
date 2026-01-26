
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegressionLoss:
    """
    Basic regression loss for field/drift/mean-map training.

    In DSB/IPF caches typically you regress:
      - target = (x_prev - x_next)  OR  (x_prev) depending on parametrization
    We keep it generic: predict -> target.
    """
    kind: str = "mse"  # "mse" | "huber"
    huber_delta: float = 1.0
    reduction: str = "mean"  # "mean" | "sum"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.kind == "mse":
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.kind == "huber":
            loss = F.huber_loss(pred, target, reduction="none", delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss kind: {self.kind}")

        # loss shape: (batch, dim)
        loss = loss.mean(dim=1)  # per-sample

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unknown reduction: {self.reduction}")

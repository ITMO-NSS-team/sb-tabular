
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GaussianReference:
    """
    Simple Gaussian reference distribution/process endpoints.

    In DSB/IPF practice for tabular:
      - terminal distribution at t=T is often standard Gaussian
      - initial distribution at t=0 is data distribution

    This class provides sampling for the "noise/prior" endpoint.
    """
    dim: int
    mean: float = 0.0
    std: float = 1.0
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def sample(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        if n <= 0:
            raise ValueError("n must be positive")
        g = torch.Generator(device=str(self.device) if self.device is not None else "cpu")
        if seed is not None:
            g.manual_seed(int(seed))
        dev = self.device or torch.device("cpu")
        x = torch.randn((n, self.dim), generator=g, device=dev, dtype=self.dtype)
        return x * self.std + self.mean


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass(frozen=True)
class TimeGrid:
    """
    Discrete time grid for SB / DSB-like solvers.

    We store:
      - K steps (indices 0..K-1)
      - per-step step sizes gamma[k] > 0
      - cumulative time t[k] = sum_{j<=k} gamma[j]
      - total time T = sum_k gamma[k]

    Conventions:
      - "forward" direction typically moves from t=0 to t=T (increasing k)
      - "backward" direction typically moves from t=T to t=0 (decreasing k)
    """
    num_steps: int
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: Literal["linear", "geom"] = "geom"
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        if self.gamma_min <= 0 or self.gamma_max <= 0:
            raise ValueError("gamma_min/gamma_max must be positive")
        if self.gamma_min > self.gamma_max:
            raise ValueError("gamma_min must be <= gamma_max")

    def gammas(self) -> torch.Tensor:
        dev = self.device or torch.device("cpu")
        if self.schedule == "linear":
            g = torch.linspace(self.gamma_min, self.gamma_max, self.num_steps, device=dev, dtype=self.dtype)
        elif self.schedule == "geom":
            g = torch.logspace(
                torch.log10(torch.tensor(self.gamma_min, device=dev, dtype=self.dtype)),
                torch.log10(torch.tensor(self.gamma_max, device=dev, dtype=self.dtype)),
                self.num_steps,
                device=dev,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        return g

    def times(self) -> torch.Tensor:
        g = self.gammas()
        return torch.cumsum(g, dim=0)

    def total_time(self) -> torch.Tensor:
        return self.gammas().sum()

    def k_to_t(self, k: torch.Tensor) -> torch.Tensor:
        """
        Map integer step indices (0..K-1) to time values t[k].
        """
        t = self.times()
        return t[k]

    def remaining_time_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Remaining time until T for time values in [0, T].
        """
        T = self.total_time()
        return T - t

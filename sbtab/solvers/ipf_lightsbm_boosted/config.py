from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LightSBMBoostedConfig:
    n_pairs: int = 100_000

    epsilon: float = 0.1
    safe_t: float = 0.01

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    n_jobs: int = -1
    lgbm_extra: Dict[str, Any] = field(default_factory=dict)

    n_euler_steps: int = 200

    seed: int = 42

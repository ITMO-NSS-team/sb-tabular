from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame]


@dataclass
class BaselineFitInfo:
    n_rows: int
    n_cols: int
    columns: list[str]


class BaselineGenerativeModel(ABC):
    """
    Unified API for baseline generative models.

    - fit(data): train on provided data (no splitting inside)
    - sample(n): generate n samples in the same schema/column order
    """

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.columns_: Optional[list[str]] = None
        self.fit_info_: Optional[BaselineFitInfo] = None

    @abstractmethod
    def fit(self, data: ArrayLike, **kwargs: Any) -> "BaselineGenerativeModel":
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> ArrayLike:
        raise NotImplementedError

    def get_fit_info(self) -> BaselineFitInfo:
        if self.fit_info_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.fit_info_

    # ----- helpers -----

    def _to_numpy_and_columns(self, data: ArrayLike) -> tuple[np.ndarray, Optional[list[str]]]:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(copy=True), list(data.columns)
        if isinstance(data, np.ndarray):
            return data, None
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _format_output(self, x: np.ndarray) -> ArrayLike:
        if self.columns_ is not None:
            return pd.DataFrame(x, columns=self.columns_)
        return x

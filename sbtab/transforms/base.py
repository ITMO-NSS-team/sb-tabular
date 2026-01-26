
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import pandas as pd

from sbtab.data.schema import TabularSchema


@dataclass
class TransformState:
    """
    Minimal serializable state for a single transform.

    NOTE:
      - `params` must be JSON-serializable if you want JSON persistence.
      - For numpy arrays, store lists.
    """
    name: str
    params: Dict[str, Any]


@runtime_checkable
class BaseTransform(Protocol):
    """
    A transform in this project is:
      - DataFrame in -> DataFrame out
      - optionally fitted (stateless transforms can ignore fit)
      - optionally invertible

    The DataModule relies on:
      - being able to call `.transform(df)` before `.fit(...)` for missing-removal
        (DropMissingRows supports this).
      - cloning via `get_state()/from_state()` for fold-wise refitting.
    """

    name: str

    def requires_fit(self) -> bool:
        """
        True if transform must be fitted before transform().
        (e.g., StandardScaler needs fit; DropMissingRows does not.)
        """
        ...

    def is_invertible(self) -> bool:
        ...

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "BaseTransform":
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If not invertible, should be identity (or raise).
        We use identity by convention to simplify pipelines.
        """
        ...

    def get_state(self) -> TransformState:
        ...

    @classmethod
    def from_state(cls, state: TransformState) -> "BaseTransform":
        ...


@dataclass
class TransformPipelineState:
    """Serializable state for a pipeline."""
    transforms: List[TransformState]

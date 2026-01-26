
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from .base import TransformState
from sbtab.data.schema import TabularSchema


@dataclass
class DropMissingRows:
   
    name: str = "drop_missing_rows"
    subset_cols: Optional[List[str]] = None

    # diagnostics (populated on last transform call)
    kept_index_: Optional[pd.Index] = None
    dropped_index_: Optional[pd.Index] = None

    def requires_fit(self) -> bool:
        return False

    def is_invertible(self) -> bool:
        return False

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "DropMissingRows":
        # Stateless: nothing to fit.
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        subset = self.subset_cols
        if subset is None:
            mask_keep = ~df.isna().any(axis=1)
        else:
            mask_keep = ~df[subset].isna().any(axis=1)

        self.kept_index_ = df.index[mask_keep]
        self.dropped_index_ = df.index[~mask_keep]
        return df.loc[self.kept_index_].copy()

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Not invertible; identity by convention.
        return df

    def get_state(self) -> TransformState:
        return TransformState(
            name=self.name,
            params={"subset_cols": self.subset_cols},
        )

    @classmethod
    def from_state(cls, state: TransformState) -> "DropMissingRows":
        return cls(subset_cols=state.params.get("subset_cols"))

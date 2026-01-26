
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import pandas as pd


@dataclass(frozen=True)
class TabularSchema:
    """
    Schema for *fully continuous* tabular data.

    Assumptions:
      - All features are continuous (float-like).
      - Target (if present) is also continuous.
    """
    feature_cols: List[str]
    target_col: Optional[str] = None
    id_col: Optional[str] = None  # optional row identifier column

    @property
    def continuous_cols(self) -> List[str]:
        # Fully continuous setting: all features are continuous.
        return list(self.feature_cols)

    @property
    def all_cols(self) -> List[str]:
        cols: List[str] = []
        if self.id_col is not None:
            cols.append(self.id_col)
        cols.extend(self.feature_cols)
        if self.target_col is not None:
            cols.append(self.target_col)
        return cols

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        # In this project phase we assume continuous data; fail early if obviously non-numeric.
        to_check = self.feature_cols + ([self.target_col] if self.target_col else [])
        non_numeric = [c for c in to_check if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise TypeError(
                "Non-numeric columns found in continuous-only setting: "
                f"{non_numeric}. Please cast/encode beforehand."
            )

    @classmethod
    def infer_from_dataframe(
        cls,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        id_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        drop_non_numeric: bool = False,
    ) -> "TabularSchema":
        """
        Infer schema from a DataFrame.

        If feature_cols is None:
          - uses all columns except target_col and id_col
          - optionally drops non-numeric columns if drop_non_numeric=True
        """
        cols = list(df.columns)

        if id_col is not None and id_col not in cols:
            raise ValueError(f"id_col='{id_col}' not found in df.columns")
        if target_col is not None and target_col not in cols:
            raise ValueError(f"target_col='{target_col}' not found in df.columns")

        if feature_cols is None:
            exclude = set([c for c in [id_col, target_col] if c is not None])
            feats = [c for c in cols if c not in exclude]
        else:
            feats = list(feature_cols)

        if drop_non_numeric:
            feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]

        schema = cls(feature_cols=feats, target_col=target_col, id_col=id_col)
        schema.validate(df)
        return schema

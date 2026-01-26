
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

from .base import TransformState
from sbtab.data.schema import TabularSchema


@dataclass
class ContinuousStandardScaler:
  
    name: str = "continuous_standard_scaler"
    eps: float = 1e-12

    # fitted state
    fitted_: bool = False
    continuous_cols_: List[str] = field(default_factory=list)
    means_: Dict[str, float] = field(default_factory=dict)
    stds_: Dict[str, float] = field(default_factory=dict)

    def requires_fit(self) -> bool:
        return True

    def is_invertible(self) -> bool:
        return True

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "ContinuousStandardScaler":
        cols = list(schema.continuous_cols)
        if not cols:
            raise ValueError("No continuous columns in schema.")

        x = df[cols].astype(float)
        means = x.mean(axis=0, skipna=True)
        stds = x.std(axis=0, ddof=0, skipna=True)

        self.continuous_cols_ = cols
        self.means_ = {c: float(means[c]) for c in cols}
        self.stds_ = {c: float(max(float(stds[c]), self.eps)) for c in cols}
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("ContinuousStandardScaler must be fitted before transform().")

        out = df.copy()
        for c in self.continuous_cols_:
            out[c] = (out[c].astype(float) - self.means_[c]) / self.stds_[c]
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("ContinuousStandardScaler must be fitted before inverse_transform().")

        out = df.copy()
        for c in self.continuous_cols_:
            out[c] = out[c].astype(float) * self.stds_[c] + self.means_[c]
        return out

    def get_state(self) -> TransformState:
        if not self.fitted_:
            # Important: DataModule clones pipelines *before* fitting fold-wise, so
            # serialization should still work for unfitted scalers.
            return TransformState(
                name=self.name,
                params={
                    "eps": self.eps,
                    "fitted": False,
                    "continuous_cols": [],
                    "means": {},
                    "stds": {},
                },
            )

        return TransformState(
            name=self.name,
            params={
                "eps": self.eps,
                "fitted": True,
                "continuous_cols": self.continuous_cols_,
                "means": self.means_,
                "stds": self.stds_,
            },
        )

    @classmethod
    def from_state(cls, state: TransformState) -> "ContinuousStandardScaler":
        obj = cls(eps=float(state.params.get("eps", 1e-12)))
        fitted = bool(state.params.get("fitted", False))
        obj.fitted_ = fitted
        if fitted:
            obj.continuous_cols_ = list(state.params.get("continuous_cols", []))
            obj.means_ = dict(state.params.get("means", {}))
            obj.stds_ = dict(state.params.get("stds", {}))
        return obj

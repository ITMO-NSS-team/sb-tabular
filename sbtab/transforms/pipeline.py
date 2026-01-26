
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Type, Dict
import pandas as pd

from .base import BaseTransform, TransformPipelineState, TransformState
from sbtab.data.schema import TabularSchema
from .missing import DropMissingRows
from .continuous import ContinuousStandardScaler


# Simple registry for (de)serialization of pipeline components.
# Add new transforms here when you expand the project.
_TRANSFORM_REGISTRY: Dict[str, Type[BaseTransform]] = {
    "drop_missing_rows": DropMissingRows,
    "continuous_standard_scaler": ContinuousStandardScaler,
}


@dataclass
class TransformPipeline:
    """
    Sequential pipeline of transforms.

    Requirements derived from DataModule logic:
      - `.transform(df)` must work even if some transforms are not fitted,
        as long as those transforms do not require fit.
        (We therefore SKIP unfitted transforms that require fit in transform().)

      - `.fit(df, schema)` fits transforms in order; by convention:
          * missing-removal happens first (stateless)
          * scaler is fitted on the df after missing-removal
    """
    transforms: List[BaseTransform] = field(default_factory=list)
    name: str = "transform_pipeline"

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "TransformPipeline":
        x = df
        for tr in self.transforms:
            tr.fit(x, schema)
            x = tr.transform(x)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df
        for tr in self.transforms:
            if tr.requires_fit() and not _is_fitted(tr):
                # Critical behavior: allow DataModule to call transform() pre-fit
                # to perform global missing removal. We skip fitted-only transforms.
                continue
            x = tr.transform(x)
        return x

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df
        for tr in reversed(self.transforms):
            # For non-invertible transforms (drop rows), inverse is identity by convention.
            x = tr.inverse_transform(x)
        return x

    def get_state(self) -> TransformPipelineState:
        return TransformPipelineState(transforms=[tr.get_state() for tr in self.transforms])

    @classmethod
    def from_state(cls, state: TransformPipelineState) -> "TransformPipeline":
        trs: List[BaseTransform] = []
        for tr_state in state.transforms:
            if tr_state.name not in _TRANSFORM_REGISTRY:
                raise KeyError(f"Unknown transform '{tr_state.name}'. Register it in _TRANSFORM_REGISTRY.")
            tr_cls = _TRANSFORM_REGISTRY[tr_state.name]
            trs.append(tr_cls.from_state(tr_state))  # type: ignore[arg-type]
        return cls(transforms=trs)

    @classmethod
    def default_continuous_dropna(cls) -> "TransformPipeline":
        """
        Default pipeline for current project assumptions:
          1) drop missing rows
          2) standard scale continuous columns
        """
        return cls(transforms=[DropMissingRows(), ContinuousStandardScaler()])


def _is_fitted(tr: BaseTransform) -> bool:
    # Convention: fitted transforms may expose "fitted_" attribute.
    return bool(getattr(tr, "fitted_", True))

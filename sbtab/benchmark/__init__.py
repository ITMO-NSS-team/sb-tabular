"""Unified, model-independent benchmark data boundary.

The package is intentionally independent of the legacy ``sbtab.data``,
``sbtab.transforms``, and ``sbtab.experiments`` orchestration paths.
"""

from __future__ import annotations

from sbtab.benchmark.contracts import (
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    DiscreteView,
    InputSpec,
    PreparedSchema,
    PreparedTable,
    StateColumn,
    TabularDataset,
    TaskType,
)
from sbtab.benchmark.missing import (
    ClassCount,
    MissingPolicy,
    MissingPolicyResult,
    MissingReport,
    MissingValuesError,
    apply_missing_policy,
)
from sbtab.benchmark.splitting import (
    FoldSplit,
    HoldoutConfig,
    HoldoutSplit,
    KFoldConfig,
    SplitConfig,
    StratifiedHoldoutConfig,
    StratifiedKFoldConfig,
    make_holdout,
    make_splits,
)
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_input_spec,
    validate_prepared_table,
    validate_tabular_dataset,
)

__all__ = [
    "CategoricalView",
    "ClassCount",
    "ColumnKind",
    "ColumnSpec",
    "ContinuousView",
    "ContractViolation",
    "DiscreteView",
    "FoldSplit",
    "HoldoutConfig",
    "HoldoutSplit",
    "InputSpec",
    "KFoldConfig",
    "MissingPolicy",
    "MissingPolicyResult",
    "MissingReport",
    "MissingValuesError",
    "PreparedSchema",
    "PreparedTable",
    "SplitConfig",
    "StateColumn",
    "StratifiedHoldoutConfig",
    "StratifiedKFoldConfig",
    "TabularDataset",
    "TaskType",
    "apply_missing_policy",
    "make_holdout",
    "make_splits",
    "validate_input_spec",
    "validate_prepared_table",
    "validate_tabular_dataset",
]

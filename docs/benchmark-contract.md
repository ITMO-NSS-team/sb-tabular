# Unified benchmark data contract

Status: reviewable benchmark foundation. This document covers the public data
boundary, validation, common missing-value and split policies, the fold-local
codec, and the thin model-adapter boundary. Runners and evaluation are
introduced in separate changes.

## Purpose

The repository currently has model-specific experiment paths with different
assumptions about column types, preprocessing, targets, and native input
layouts. The unified benchmark needs one semantic data boundary before those
paths can be migrated safely.

The boundary separates two questions:

1. What does each raw column mean?
2. In which semantic representation does a model consume that column group?

It deliberately does not standardize a model's tensor layout, backend dtype,
device, loader, or temporary `X`/`y` call signature. Those are adapter-owned
integration details, not dataset semantics.

## Contract objects

### `ColumnSpec`

`ColumnSpec` declares one modeled raw column:

- `name` is its exact DataFrame label;
- `kind` is `CONTINUOUS`, `DISCRETE`, or `CATEGORICAL`;
- `ordered_values` optionally declares a complete ordinal domain for a
  categorical column.

Column kind is explicit. Shared code and adapters must not infer it from pandas
dtype, observed cardinality, or a dataset name.

Continuous columns have real-valued support. Discrete columns have finite
numeric support with meaningful order. Categorical columns are nominal unless
`ordered_values` gives an explicit semantic order.

Numeric discrete support is always ordered by ascending numeric value.
`ordered_values` cannot override that order; the field exists only for ordinal
categorical values such as `("low", "medium", "high")`.

### `TabularDataset`

`TabularDataset` is the only public raw-dataset object in the new benchmark. It
contains:

- the raw DataFrame;
- ordered `ColumnSpec` entries for every modeled column;
- an optional target and its task;
- an optional identifier.

The `columns` tuple defines canonical generated-table order. The raw frame must
contain exactly those modeled columns plus, optionally, the declared
identifier. Undeclared columns are rejected instead of silently dropped.

The target remains an ordinary modeled column. It is not removed from the
table or encoded as a separate input mode. If a native model API requires a
separate `y`, its adapter may extract and later reassemble it.

The target kind must agree with its declared utility task:

- classification accepts categorical or numeric discrete targets;
- regression accepts continuous or numeric discrete targets.

A numeric discrete target is valid for either task because `task`, rather than
storage dtype, states whether its finite numeric values are class labels or a
regression response.

An identifier is never modeled. A future runner may retain it for audit or
reconstruct output identifiers after generation, but it must not pass training
identifiers to a generative model.

### `InputSpec`

`InputSpec` is the complete shared request made by one model family. It has
exactly three fields:

| Semantic group | Supported views |
| --- | --- |
| continuous | `RAW`, `STANDARD`, `UNSUPPORTED` |
| discrete | `RAW_VALUES`, `FINITE_STATE_CODES`, `UNSUPPORTED` |
| categorical | `RAW_VALUES`, `FINITE_STATE_CODES`, `UNSUPPORTED` |

These fields describe semantic values prepared by a future fold-local codec.
They do not describe DataFrames versus arrays, tensor dtypes, devices, target
extraction, missing-value support, or model hyperparameters.

`UNSUPPORTED` means that a model rejects datasets with a non-empty group of
that kind. It is an explicit limitation, not a request for an automatic
fallback.

### `PreparedSchema`

`PreparedSchema` describes a canonical prepared table:

- `column_order` includes every modeled column and the target;
- the three semantic column tuples partition `column_order` exactly;
- `target_col` and `task_type` preserve evaluation semantics;
- `state_columns` records cardinality and ordering for named finite-state
  columns.

State metadata is keyed by column name. Cardinality and ordering must never be
inferred from a model-specific array position or replaced with a shared maximum
cardinality.

The `state_columns` mapping is copied and made read-only during construction.
This prevents later mutation of a caller-owned dictionary from changing an
already constructed schema. Code that serializes artifacts must explicitly
copy it with `dict(schema.state_columns)` instead of assuming that a mutable
dictionary remains attached to the frozen schema.

### `PreparedTable`

`PreparedTable` is the future adapter input and output. It consists of a
DataFrame and its `PreparedSchema`.

The frame:

- follows `schema.column_order` exactly;
- contains the target when declared;
- excludes the raw identifier;
- contains no missing or non-finite numeric values;
- uses integer codes in `[0, cardinality)` for encoded state columns.

Generated invalid states are rejected. Shared code must not clip, round, pad,
replace, or silently drop them to make a model run succeed.

## Validation ownership

`sbtab/benchmark/validation.py` validates the boundary without changing data:

- dataset validation checks raw declarations and observed value types;
- input validation accepts only the defined semantic enums;
- prepared-schema validation checks order, partitioning, target semantics, and
  complete finite-state metadata;
- prepared-table validation checks physical DataFrame values against the
  attached schema.

Missing-value filtering and learned transforms are not validation operations.
The next section assigns filtering to one common pre-split policy. Learned
transforms remain the responsibility of a later fold-local codec and must be
applied uniformly across models.

## Missing values before splitting

`sbtab.benchmark.missing` owns one experiment-wide policy over the raw modeled
table. The caller selects an enum value explicitly:

- `ERROR` preserves all rows and raises `MissingValuesError` when any modeled
  value is missing;
- `COMPLETE_CASE` removes a row when any modeled column, including target, is
  missing.

The optional identifier is ignored by both policies because it never enters a
model. `COMPLETE_CASE` may therefore retain a row whose identifier is missing.
No adapter or model receives the policy or may add its own fallback.

`apply_missing_policy` returns a `MissingPolicyResult` containing the retained
`TabularDataset` and a `MissingReport`. Under `ERROR`, the same report is
attached to the exception. It records:

- row counts before and after policy application;
- dropped row count and source-row fraction;
- pre-policy missing counts for every modeled column in canonical order;
- raw classification-target counts before and after filtering when applicable.

The official v1 comparison will select `COMPLETE_CASE` once before creating
common splits. `ERROR` is the intended safe default when benchmark
configuration is introduced. This module itself has no implicit default:
callers must pass the policy. Imputation and model-native missing handling are
not v1 enum values and require a separate contract decision.

The policy layer itself does not choose train and held-out rows. The splitting
component below consumes its post-policy dataset, so every model sees identical
source rows before any fold-local preprocessing is fitted.

## Benchmark-owned splitting

`sbtab.benchmark.splitting` produces immutable positional row partitions. It
does not fit preprocessing, inspect model requirements, or expose held-out raw
rows to a generator. Positions refer to the post-policy frame rather than its
pandas index labels, so custom or duplicated index labels cannot change split
membership.

Two split families have different purposes:

- `HoldoutConfig` and `StratifiedHoldoutConfig` create the train/validation
  partition used to select model hyperparameters. Their reference defaults are
  an 80/20 split with seed `5`.
- `KFoldConfig` and `StratifiedKFoldConfig` create the common final comparison
  folds. Their reference defaults are five shuffled folds with seed `42`.

Stratified variants are valid only for classification with a declared
discrete or categorical target. They read raw target values solely to assign
rows while preserving class representation. The target remains an ordinary
modeled column in each training partition; it is not passed separately through
the shared contract. Each observed class must have enough rows to appear in
every requested K-fold partition, and a stratified holdout must leave room for
each class in both partitions.

Splitters reject modeled missing values. Callers must first apply the single
experiment-wide `MissingPolicy`; a model or adapter cannot substitute its own
row filtering. Splitting also precedes every learned transform. Consequently,
the codec below fits only on the returned training positions, and held-out
categories or numeric distributions cannot influence its fitted state.

## Fold-local model codec

`compile_codec(dataset, input_spec)` validates dataset semantics and creates an
unfitted, single-use `ModelCodec`. Its `fit_transform(train_raw)` method learns
generic preprocessing from one raw training partition and returns the
canonical `PreparedTable` accepted by an adapter. The codec does not retain the
complete dataset frame and deliberately exposes no method for transforming
held-out rows.

The codec owns all model-independent learned transforms requested by
`InputSpec`:

- `STANDARD` uses the training-partition population mean and standard
  deviation for each continuous column; constant columns use scale `1`;
- `FINITE_STATE_CODES` builds a separate train-observed codebook for each
  discrete or categorical column;
- raw finite-state views preserve values but still record their training
  support so unseen generated values fail during decoding.

Discrete codebooks follow ascending numeric order. Ordered categorical
codebooks follow the declared order restricted to values observed in the
training partition. Nominal categorical codebooks follow first-observed train
order. Cardinality and ordering metadata are keyed by column name in the
prepared schema.

`inverse_transform(sample)` accepts a validated model sample carrying the
exact schema object produced by that codec. It reverses learned transforms to
raw semantic values and rejects invalid state codes or raw finite values absent
from train support. It does not clip, round, impute, or coerce generated values
to mimic the original pandas storage dtype. Identifier columns remain outside
both prepared and decoded modeled tables.

## Thin model-adapter boundary

Every model integration implements the runtime-checkable `ModelAdapter`
protocol:

- `name` is a stable artifact label;
- `input_spec` declares only the three semantic views;
- `fit(train, context)` consumes one complete prepared training table;
- `sample(n, seed)` returns all modeled columns in a `PreparedTable` carrying
  the same fold schema.

`RunContext` contains only model-independent runtime controls: run and fold
identifiers, training seed, device string, and a fold-specific artifact path.
Native arrays or tensors, dtypes, layouts, loaders, temporary target
extraction, and model hyperparameters remain adapter-owned details. An adapter
does not split data, fit generic preprocessing, decode values, or calculate
metrics.

## Dependency boundary

The new `sbtab.benchmark` package must not import the legacy orchestration APIs
under:

- `sbtab.data`;
- `sbtab.transforms`;
- `sbtab.experiments`.

Those modules remain useful migration evidence, but extending them would bind
the new contract to the assumptions it is intended to replace. An AST-based
test checks both absolute and relative imports without importing their targets.
The reverse boundary is equally strict: native `bridge`, `models`, and
`solvers` code must not import the higher-level `benchmark` or `evaluation`
packages.

## Verification

From the repository root, run the tests owned by this contract:

```bash
python -m unittest \
  tests.benchmark.test_contracts \
  tests.benchmark.test_import_boundaries \
  tests.benchmark.test_missing \
  tests.benchmark.test_splitting \
  tests.benchmark.test_adapter \
  tests.benchmark.test_codec
```

The tests cover malformed declarations, target/task and identifier rules,
semantic partitions, finite-state ranges, timestamp category identity,
canonical order, both dependency directions, uniform pre-split missing
handling, deterministic row partitions, and target stratification constraints.
Codec tests additionally cover train-only state, reversible semantic views,
schema identity, unsupported groups, identifiers, and invalid generated
states. Adapter tests cover structural metadata and explicit runtime controls.

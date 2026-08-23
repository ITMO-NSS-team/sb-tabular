"""Crash-safe fold checkpoints for final cross-validation generation.

The artifact store fingerprints the deterministic post-policy plan before any
model is fitted. Each completed fold is then committed as an atomic directory:
the synthetic table is written first and ``fold.json`` is the commit marker.
A repeated invocation loads committed folds after verifying their checksums and
executes only the missing folds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import cast
from uuid import uuid4

import numpy as np
import pandas as pd

from sbtab.benchmark.adapter import ModelAdapter, validate_adapter_definition
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.runner import (
    BenchmarkConfig,
    CrossValidationPlan,
    CrossValidationResult,
    FoldExecution,
    FoldResult,
    assemble_cross_validation,
    prepare_cross_validation,
    run_cross_validation_fold,
)
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_tabular_dataset,
)


GENERATION_ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class PersistedCrossValidation:
    """Completed generation result and evidence about restart behavior."""

    result: CrossValidationResult
    manifest_path: Path
    resumed_fold_ids: tuple[int, ...]
    generated_fold_ids: tuple[int, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except FileNotFoundError as error:
        raise ContractViolation(f"Artifact file does not exist: {path}.") from error


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_value(value: object) -> object:
    """Convert immutable benchmark metadata to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractViolation(
                "Artifact metadata must not contain non-finite floats."
            )
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"type": "timedelta", "value": value.isoformat()}
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractViolation(
                    "Artifact metadata mapping keys must be strings."
                )
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ContractViolation(
        "Artifact metadata contains unsupported value "
        f"{value!r} of type {type(value).__name__}."
    )


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if np.ndim(missing) == 0 else False
    except (TypeError, ValueError):
        return False


def _encode_scalar(value: object) -> dict[str, object]:
    """Encode supported pandas scalars without ambiguous JSON coercion."""

    if _is_missing_scalar(value):
        return {"type": "null"}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractViolation(
                "Artifact tables must not contain non-finite floats."
            )
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"type": "timedelta", "value": value.isoformat()}
    raise ContractViolation(
        "Artifact tables support null, bool, integer, finite float, string, "
        "pandas Timestamp, and pandas Timedelta scalars; got "
        f"{value!r} ({type(value).__name__})."
    )


def _decode_scalar(payload: object) -> object:
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        raise ContractViolation("Artifact table contains an invalid scalar record.")
    scalar_type = payload["type"]
    if scalar_type == "null":
        return None
    value = payload.get("value")
    if scalar_type == "bool" and isinstance(value, bool):
        return value
    if scalar_type == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if scalar_type == "float" and isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    if scalar_type == "str" and isinstance(value, str):
        return value
    if scalar_type == "timestamp" and isinstance(value, str):
        return pd.Timestamp(value)
    if scalar_type == "timedelta" and isinstance(value, str):
        return pd.Timedelta(value)
    raise ContractViolation(
        f"Artifact table contains invalid {scalar_type!r} scalar payload."
    )


def _table_payload(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": frame.columns.tolist(),
        "rows": [
            [_encode_scalar(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def _table_from_payload(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ContractViolation("Artifact table root must be an object.")
    columns = payload.get("columns")
    rows = payload.get("rows")
    if (
        not isinstance(columns, list)
        or not all(isinstance(name, str) for name in columns)
        or not isinstance(rows, list)
    ):
        raise ContractViolation("Artifact table columns or rows are invalid.")
    decoded_rows: list[list[object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            raise ContractViolation("Artifact table row width is invalid.")
        decoded_rows.append([_decode_scalar(value) for value in row])
    return pd.DataFrame(decoded_rows, columns=cast(list[str], columns))


def _read_json(path: Path, *, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractViolation(f"{label} does not exist: {path}.") from error
    except json.JSONDecodeError as error:
        raise ContractViolation(f"{label} is not valid JSON: {path}.") from error


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dataset_manifest(dataset: TabularDataset) -> dict[str, object]:
    return {
        "name": dataset.name,
        "target": dataset.target,
        "task": _json_value(dataset.task),
        "identifier": dataset.identifier,
        "raw_columns": dataset.frame.columns.tolist(),
        "modeled_columns": [
            {
                "name": column.name,
                "kind": column.kind.value,
                "ordered_values": _json_value(column.ordered_values),
            }
            for column in dataset.columns
        ],
        "rows": len(dataset.frame),
    }


def _config_manifest(config: BenchmarkConfig) -> dict[str, object]:
    return {
        "split": {
            "type": type(config.split).__name__,
            **{
                field.name: _json_value(getattr(config.split, field.name))
                for field in fields(config.split)
            },
        },
        "missing_policy": config.missing_policy.value,
        "run_id": config.run_id,
        "training_seed": config.training_seed,
        "sample_seed": config.sample_seed,
        "device": config.device,
        "artifact_dir": str(config.artifact_dir),
    }


def _plan_core(
    plan: CrossValidationPlan,
    *,
    real_path: str,
    real_sha256: str,
) -> dict[str, object]:
    return {
        "artifact_type": "cross_validation_plan",
        "artifact_version": GENERATION_ARTIFACT_VERSION,
        "dataset": _dataset_manifest(plan.dataset),
        "config": _config_manifest(plan.config),
        "missing_report": _json_value(plan.missing_report),
        "splits": [_json_value(split) for split in plan.splits],
        "real_path": real_path,
        "real_sha256": real_sha256,
    }


def _plan_fingerprint(plan: CrossValidationPlan) -> str:
    real_digest = _sha256_bytes(_json_bytes(_table_payload(plan.dataset.frame)))
    core = _plan_core(
        plan,
        real_path="real-post-policy.json",
        real_sha256=real_digest,
    )
    return _sha256_bytes(_json_bytes(core))


def cross_validation_result_fingerprint(result: CrossValidationResult) -> str:
    """Return a deterministic identity for decoded generation result bytes."""

    if not isinstance(result, CrossValidationResult):
        raise ContractViolation("result must be CrossValidationResult.")
    plan = CrossValidationPlan(
        config=result.config,
        dataset=result.dataset,
        missing_report=result.missing_report,
        splits=tuple(fold.split for fold in result.folds),
    )
    payload = {
        "plan_fingerprint": _plan_fingerprint(plan),
        "adapter_name": result.adapter_name,
        "folds": [
            {
                "fold_id": fold.split.fold_id,
                "synthetic_sha256": _sha256_bytes(
                    _json_bytes(_table_payload(fold.synthetic_raw))
                ),
                "fit_seconds": fold.fit_seconds,
                "sample_seconds": fold.sample_seconds,
            }
            for fold in result.folds
        ],
    }
    return _sha256_bytes(_json_bytes(payload))


def _validate_synthetic(plan: CrossValidationPlan, frame: pd.DataFrame) -> None:
    validate_tabular_dataset(
        TabularDataset(
            name=f"{plan.dataset.name}:artifact-synthetic",
            frame=frame,
            columns=plan.dataset.columns,
            target=plan.dataset.target,
            task=plan.dataset.task,
        )
    )


def _modeled_partition(
    dataset: TabularDataset,
    positions: tuple[int, ...],
) -> pd.DataFrame:
    return (
        dataset.frame.iloc[list(positions)]
        .loc[:, list(dataset.column_order)]
        .reset_index(drop=True)
        .copy()
    )


class GenerationArtifactStore:
    """Verified local checkpoint store bound to one cross-validation plan."""

    def __init__(
        self,
        plan: CrossValidationPlan,
        output_dir: Path,
        plan_payload: dict[str, object],
    ) -> None:
        self.plan = plan
        self.output_dir = output_dir
        self.plan_payload = plan_payload

    @property
    def fingerprint(self) -> str:
        value = self.plan_payload.get("fingerprint")
        if not isinstance(value, str):
            raise ContractViolation("Generation plan fingerprint is invalid.")
        return value

    @classmethod
    def open_or_create(
        cls,
        plan: CrossValidationPlan,
        output_dir: Path,
    ) -> GenerationArtifactStore:
        """Create a plan atomically or verify an existing plan for resume."""

        if not isinstance(plan, CrossValidationPlan):
            raise ContractViolation("plan must be CrossValidationPlan.")
        if not isinstance(output_dir, Path):
            raise ContractViolation("output_dir must be pathlib.Path.")
        plan_path = output_dir / "plan.json"
        if output_dir.exists():
            payload = _read_json(plan_path, label="Generation plan")
            if not isinstance(payload, dict):
                raise ContractViolation("Generation plan root must be an object.")
            cls._verify_existing_plan(plan, output_dir, payload)
            return cls(plan, output_dir, payload)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_dir.with_name(
            f".{output_dir.name}.initializing-{uuid4().hex}"
        )
        temporary.mkdir()
        try:
            real_path = temporary / "real-post-policy.json"
            real_bytes = _json_bytes(_table_payload(plan.dataset.frame))
            real_path.write_bytes(real_bytes)
            core = _plan_core(
                plan,
                real_path=real_path.name,
                real_sha256=_sha256_bytes(real_bytes),
            )
            payload = {
                **core,
                "fingerprint": _sha256_bytes(_json_bytes(core)),
            }
            (temporary / "plan.json").write_bytes(_json_bytes(payload))
            temporary.replace(output_dir)
        except FileExistsError as error:
            raise ContractViolation(
                f"Artifact directory was created concurrently: {output_dir}."
            ) from error
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return cls(plan, output_dir, payload)

    @staticmethod
    def _verify_existing_plan(
        plan: CrossValidationPlan,
        output_dir: Path,
        payload: dict[str, object],
    ) -> None:
        if payload.get("artifact_type") != "cross_validation_plan" or (
            payload.get("artifact_version") != GENERATION_ARTIFACT_VERSION
        ):
            raise ContractViolation("Existing artifact plan has an unsupported type or version.")
        real_name = payload.get("real_path")
        expected_real_digest = payload.get("real_sha256")
        if not isinstance(real_name, str) or not isinstance(expected_real_digest, str):
            raise ContractViolation("Existing artifact plan has invalid real-table metadata.")
        real_path = output_dir / real_name
        if _sha256_file(real_path) != expected_real_digest:
            raise ContractViolation("Stored post-policy real table checksum does not match.")
        current_real_digest = _sha256_bytes(
            _json_bytes(_table_payload(plan.dataset.frame))
        )
        if current_real_digest != expected_real_digest:
            raise ContractViolation(
                "Existing artifact plan does not match current dataset values."
            )
        core = _plan_core(
            plan,
            real_path=real_name,
            real_sha256=expected_real_digest,
        )
        expected_fingerprint = _sha256_bytes(_json_bytes(core))
        if payload.get("fingerprint") != expected_fingerprint:
            raise ContractViolation(
                "Existing artifact plan does not match dataset, splits, or runtime controls."
            )

    def _planned_split(self, fold_id: int):
        for split in self.plan.splits:
            if split.fold_id == fold_id:
                return split
        raise ContractViolation(f"Fold {fold_id} is not part of the plan.")

    def load_fold(self, fold_id: int) -> FoldExecution | None:
        """Load one committed fold, returning ``None`` only when absent."""

        split = self._planned_split(fold_id)
        fold_dir = self.output_dir / f"fold-{fold_id}"
        if not fold_dir.exists():
            return None
        commit_path = fold_dir / "fold.json"
        payload = _read_json(commit_path, label=f"Fold {fold_id} commit")
        if not isinstance(payload, dict):
            raise ContractViolation(f"Fold {fold_id} commit root must be an object.")
        expected_split = _json_value(split)
        if payload.get("plan_fingerprint") != self.fingerprint or (
            payload.get("split") != expected_split
        ):
            raise ContractViolation(f"Fold {fold_id} does not match the generation plan.")
        synthetic_name = payload.get("synthetic_path")
        synthetic_digest = payload.get("synthetic_sha256")
        if not isinstance(synthetic_name, str) or not isinstance(synthetic_digest, str):
            raise ContractViolation(f"Fold {fold_id} synthetic metadata is invalid.")
        synthetic_path = fold_dir / synthetic_name
        if _sha256_file(synthetic_path) != synthetic_digest:
            raise ContractViolation(f"Fold {fold_id} synthetic checksum does not match.")
        synthetic = _table_from_payload(
            _read_json(synthetic_path, label=f"Fold {fold_id} synthetic table")
        )
        _validate_synthetic(self.plan, synthetic)
        if len(synthetic) != len(split.train_positions):
            raise ContractViolation(f"Fold {fold_id} synthetic row count is invalid.")

        adapter_name = payload.get("adapter_name")
        fit_seconds = payload.get("fit_seconds")
        sample_seconds = payload.get("sample_seconds")
        if not isinstance(adapter_name, str) or not adapter_name.strip():
            raise ContractViolation(f"Fold {fold_id} adapter name is invalid.")
        for label, value in (
            ("fit_seconds", fit_seconds),
            ("sample_seconds", sample_seconds),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ContractViolation(f"Fold {fold_id} {label} is invalid.")
        return FoldExecution(
            adapter_name=adapter_name,
            fold=FoldResult(
                split=split,
                train_raw=_modeled_partition(self.plan.dataset, split.train_positions),
                test_raw=_modeled_partition(self.plan.dataset, split.test_positions),
                synthetic_raw=synthetic,
                fit_seconds=float(fit_seconds),
                sample_seconds=float(sample_seconds),
            ),
        )

    def commit_fold(self, execution: FoldExecution) -> Path:
        """Atomically commit one newly completed fold without overwriting."""

        if not isinstance(execution, FoldExecution):
            raise ContractViolation("execution must be FoldExecution.")
        split = self._planned_split(execution.fold.split.fold_id)
        if execution.fold.split != split:
            raise ContractViolation("Fold execution membership differs from the plan.")
        _validate_synthetic(self.plan, execution.fold.synthetic_raw)
        if len(execution.fold.synthetic_raw) != len(split.train_positions):
            raise ContractViolation("Fold synthetic row count differs from the plan.")

        final_dir = self.output_dir / f"fold-{split.fold_id}"
        if final_dir.exists():
            raise ContractViolation(f"Fold {split.fold_id} is already committed.")
        temporary = self.output_dir / f".fold-{split.fold_id}.tmp-{uuid4().hex}"
        temporary.mkdir()
        try:
            synthetic_path = temporary / "synthetic.json"
            synthetic_bytes = _json_bytes(_table_payload(execution.fold.synthetic_raw))
            synthetic_path.write_bytes(synthetic_bytes)
            commit = {
                "artifact_type": "cross_validation_fold",
                "artifact_version": GENERATION_ARTIFACT_VERSION,
                "plan_fingerprint": self.fingerprint,
                "adapter_name": execution.adapter_name,
                "split": _json_value(split),
                "train_rows": len(execution.fold.train_raw),
                "test_rows": len(execution.fold.test_raw),
                "synthetic_rows": len(execution.fold.synthetic_raw),
                "fit_seconds": execution.fold.fit_seconds,
                "sample_seconds": execution.fold.sample_seconds,
                "synthetic_path": synthetic_path.name,
                "synthetic_sha256": _sha256_bytes(synthetic_bytes),
            }
            (temporary / "fold.json").write_bytes(_json_bytes(commit))
            temporary.replace(final_dir)
        except FileExistsError as error:
            raise ContractViolation(
                f"Fold {split.fold_id} was committed concurrently."
            ) from error
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return final_dir / "fold.json"

    def finalize(self, result: CrossValidationResult) -> Path:
        """Write the complete generation manifest after every fold exists."""

        if not isinstance(result, CrossValidationResult):
            raise ContractViolation("result must be CrossValidationResult.")
        fold_entries: list[dict[str, object]] = []
        loaded_executions: list[FoldExecution] = []
        for fold in result.folds:
            loaded = self.load_fold(fold.split.fold_id)
            if loaded is None:
                raise ContractViolation(
                    f"Fold {fold.split.fold_id} is not committed."
                )
            if loaded.adapter_name != result.adapter_name:
                raise ContractViolation(
                    f"Fold {fold.split.fold_id} adapter differs from result."
                )
            loaded_executions.append(loaded)
            commit_path = self.output_dir / f"fold-{fold.split.fold_id}" / "fold.json"
            commit = _read_json(commit_path, label=f"Fold {fold.split.fold_id} commit")
            if not isinstance(commit, dict):
                raise ContractViolation("Fold commit root must be an object.")
            fold_entries.append(
                {
                    "fold_id": fold.split.fold_id,
                    "commit_path": str(commit_path.relative_to(self.output_dir)),
                    "commit_sha256": _sha256_file(commit_path),
                    "synthetic_sha256": commit.get("synthetic_sha256"),
                }
            )
        stored_result = assemble_cross_validation(
            self.plan,
            tuple(loaded_executions),
        )
        if cross_validation_result_fingerprint(stored_result) != (
            cross_validation_result_fingerprint(result)
        ):
            raise ContractViolation(
                "Committed fold content differs from generation result."
            )
        manifest = {
            "artifact_type": "cross_validation_generation",
            "artifact_version": GENERATION_ARTIFACT_VERSION,
            "plan_path": "plan.json",
            "plan_fingerprint": self.fingerprint,
            "adapter_name": result.adapter_name,
            "dataset": result.dataset.name,
            "result_fingerprint": cross_validation_result_fingerprint(result),
            "folds": fold_entries,
        }
        manifest_path = self.output_dir / "manifest.json"
        encoded = _json_bytes(manifest)
        if manifest_path.exists():
            if manifest_path.read_bytes() != encoded:
                raise ContractViolation("Existing generation manifest differs from result.")
            return manifest_path
        _write_atomic(manifest_path, encoded)
        return manifest_path


def run_cross_validation_resumable(
    dataset: TabularDataset,
    adapter_factory: Callable[[], ModelAdapter],
    config: BenchmarkConfig,
    *,
    output_dir: Path,
) -> PersistedCrossValidation:
    """Run missing folds and return one fully verified persisted CV result."""

    if not callable(adapter_factory):
        raise ContractViolation("adapter_factory must be callable.")
    plan = prepare_cross_validation(dataset, config)
    store = GenerationArtifactStore.open_or_create(plan, output_dir)
    executions: list[FoldExecution] = []
    resumed: list[int] = []
    generated: list[int] = []
    adapters: list[ModelAdapter] = []
    for split in plan.splits:
        execution = store.load_fold(split.fold_id)
        if execution is not None:
            resumed.append(split.fold_id)
            executions.append(execution)
            continue

        adapter = adapter_factory()
        validate_adapter_definition(adapter)
        if any(adapter is previous for previous in adapters):
            raise ContractViolation(
                "adapter_factory must return a fresh adapter instance per fold."
            )
        adapters.append(adapter)
        execution = run_cross_validation_fold(plan, split, adapter)
        store.commit_fold(execution)
        generated.append(split.fold_id)
        executions.append(execution)

    result = assemble_cross_validation(plan, tuple(executions))
    return PersistedCrossValidation(
        result=result,
        manifest_path=store.finalize(result),
        resumed_fold_ids=tuple(resumed),
        generated_fold_ids=tuple(generated),
    )

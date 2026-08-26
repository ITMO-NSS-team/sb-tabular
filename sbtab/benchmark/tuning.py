"""Reusable Optuna lifecycle for model-owned benchmark tuning.

The harness owns only behavior that must be identical across model families:
study compatibility, trial accounting, one shared holdout execution, tuning
evidence, failure disposition, and reconstruction of the selected native
configuration. A model integration still owns its search space, native config
type, adapter construction, config serialization, and numerical-failure
classification.

Neither the benchmark runner nor a native model imports Optuna. Model-specific
tuning modules call :func:`run_optuna_holdout_study` and may add a later
selection stage, such as multi-seed reranking, without changing this common
Phase-A boundary.
"""

from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic
from typing import Generic, TypeVar
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd

from sbtab.benchmark.adapter import ModelAdapter, validate_adapter_definition
from sbtab.benchmark.contracts import InputSpec, TabularDataset
from sbtab.benchmark.missing import (
    ClassCount,
    MissingReport,
    apply_missing_policy,
)
from sbtab.benchmark.runner import HoldoutRunConfig, run_holdout_trial
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_input_spec,
    validate_tabular_dataset,
)
from sbtab.evaluation.tuning import TuningScore, evaluate_tuning_score


OPTUNA_HARNESS_VERSION = 1
OPTUNA_TUNING_ARTIFACT_VERSION = 1
_FINGERPRINT_ATTR = "benchmark_tuning_fingerprint"
_NATIVE_CONFIG_ATTR = "native_config"

NativeConfigT = TypeVar("NativeConfigT")


class TrialFailureAction(str, Enum):
    """Disposition assigned to one model-specific trial exception.

    ``PRUNE`` records an unfavorable but expected numerical region as a pruned
    attempt. ``FAIL`` records an expected failed attempt without treating it as
    an objective observation. ``RAISE`` records the trial as failed and
    propagates the exception, stopping the tuning invocation.
    """

    PRUNE = "prune"
    FAIL = "fail"
    RAISE = "raise"


@dataclass(frozen=True)
class TrialFailure:
    """Serializable model-owned interpretation of one trial exception.

    Parameters
    ----------
    action:
        Optuna disposition. It controls only this trial and whether the current
        invocation continues.
    kind:
        Stable machine-readable failure label, for example
        ``"non_finite_training_loss"``. It must not depend on an exception's
        localized display text.
    message:
        Human-readable diagnostic stored with the trial. It should contain
        enough native context for review but no credentials or storage URI.
    """

    action: TrialFailureAction
    kind: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, TrialFailureAction):
            raise ContractViolation(
                "TrialFailure.action must be TrialFailureAction."
            )
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ContractViolation("TrialFailure.kind must be a non-empty string.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ContractViolation(
                "TrialFailure.message must be a non-empty string."
            )


@dataclass(frozen=True)
class OptunaStudyConfig:
    """Model-independent controls for one resumable Optuna study.

    Parameters
    ----------
    run:
        Common holdout, missing policy, training/sample seeds, device, and
        logical artifact root reused by every trial. Trial-specific run labels
        and child artifact paths are derived by the harness.
    target_complete_trials:
        Desired total number of successful trials in the complete study. On
        resume this is a target, not a number of additional attempts.
    max_total_trials:
        Safety ceiling over every stored trial state, including failed and
        pruned trials. It may be increased when resuming a compatible study.
    sampler_seed:
        Seed for the default TPE sampler. Native model and sample seeds remain
        the explicit values in ``run``.
    timeout_seconds:
        Optional wall-clock budget for this invocation. A timeout leaves the
        persistent study resumable but does not manufacture a partial result.
    study_name, storage, load_if_exists:
        Optuna persistence controls. ``storage`` is consumed but never copied
        into review artifacts because a URI may contain credentials.
    protocol_version:
        Model-owned version covering search-space and failure-policy semantics.
        Increment it whenever old and new trial values must not mix.
    objective_version:
        Version of the shared score convention expected by this model study.
        Increment it when the mathematical objective changes.
    """

    run: HoldoutRunConfig
    target_complete_trials: int = 30
    max_total_trials: int = 45
    sampler_seed: int = 5
    timeout_seconds: float | None = None
    study_name: str = "benchmark-tuning"
    storage: str | None = None
    load_if_exists: bool = False
    protocol_version: int = 1
    objective_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.run, HoldoutRunConfig):
            raise ContractViolation(
                "OptunaStudyConfig.run must be HoldoutRunConfig."
            )
        for field_name in (
            "target_complete_trials",
            "max_total_trials",
            "sampler_seed",
            "protocol_version",
            "objective_version",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractViolation(
                    f"OptunaStudyConfig.{field_name} must be an integer."
                )
        if self.target_complete_trials < 1:
            raise ContractViolation("target_complete_trials must be positive.")
        if self.max_total_trials < self.target_complete_trials:
            raise ContractViolation(
                "max_total_trials cannot be below target_complete_trials."
            )
        if not 0 <= self.sampler_seed < 2**32:
            raise ContractViolation("sampler_seed must be in [0, 2**32).")
        if self.protocol_version < 1 or self.objective_version < 1:
            raise ContractViolation(
                "protocol_version and objective_version must be positive."
            )
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds,
                (int, float),
            ):
                raise ContractViolation(
                    "timeout_seconds must be a number or None."
                )
            if not np.isfinite(float(self.timeout_seconds)) or (
                self.timeout_seconds <= 0
            ):
                raise ContractViolation(
                    "timeout_seconds must be finite and positive."
                )
        if not isinstance(self.study_name, str) or not self.study_name.strip():
            raise ContractViolation("study_name must be a non-empty string.")
        if self.storage is not None and not isinstance(self.storage, str):
            raise ContractViolation("storage must be a string or None.")
        if not isinstance(self.load_if_exists, bool):
            raise ContractViolation("load_if_exists must be bool.")


@dataclass(frozen=True)
class OptunaTuningResult(Generic[NativeConfigT]):
    """Completed common study and its reconstructed best native configuration.

    Parameters
    ----------
    study:
        Native Optuna study retaining every trial state and user attribute.
    config:
        Common lifecycle controls used for this invocation.
    dataset:
        Original declared raw dataset supplied by the caller.
    model_name, input_spec:
        Adapter identity and semantic views protected by the study fingerprint.
    best_config:
        Model-native configuration reconstructed from the best completed trial.
    best_config_payload:
        JSON-compatible snapshot from which ``best_config`` was reconstructed.
    best_score:
        Minimized shared tuning objective of the best completed trial.
    fingerprint:
        Digest protecting dataset values, semantics, holdout controls, model
        identity, semantic views, sampler seed, and versioned study protocol.
    """

    study: optuna.Study
    config: OptunaStudyConfig
    dataset: TabularDataset
    model_name: str
    input_spec: InputSpec
    best_config: NativeConfigT
    best_config_payload: Mapping[str, object]
    best_score: float
    fingerprint: str


def _json_mapping(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ContractViolation(f"{label} must be a mapping.")
    try:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContractViolation(f"{label} must be JSON-compatible.") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ContractViolation(f"{label} JSON root must be an object.")
    return decoded


def _ordered_value_payload(value: object) -> dict[str, str]:
    value_type = type(value)
    return {
        "type": f"{value_type.__module__}.{value_type.__qualname__}",
        "repr": repr(value),
    }


def _study_fingerprint(
    dataset: TabularDataset,
    model_name: str,
    input_spec: InputSpec,
    config: OptunaStudyConfig,
) -> str:
    filtered = apply_missing_policy(dataset, config.run.missing_policy).dataset
    modeled = filtered.frame.loc[:, list(filtered.column_order)]
    value_hash = hashlib.sha256(
        pd.util.hash_pandas_object(modeled, index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    ).hexdigest()
    payload = {
        "harness_version": OPTUNA_HARNESS_VERSION,
        "protocol_version": config.protocol_version,
        "objective_version": config.objective_version,
        "model_name": model_name,
        "input_spec": {
            "continuous_view": input_spec.continuous_view.value,
            "discrete_view": input_spec.discrete_view.value,
            "categorical_view": input_spec.categorical_view.value,
        },
        "dataset": {
            "name": filtered.name,
            "value_sha256": value_hash,
            "storage_dtypes": [str(dtype) for dtype in modeled.dtypes],
            "columns": [
                {
                    "name": column.name,
                    "kind": column.kind.value,
                    "ordered_values": (
                        [
                            _ordered_value_payload(value)
                            for value in column.ordered_values
                        ]
                        if column.ordered_values is not None
                        else None
                    ),
                }
                for column in filtered.columns
            ],
            "target": filtered.target,
            "task": filtered.task.value if filtered.task is not None else None,
        },
        "holdout": {
            "type": type(config.run.split).__name__,
            **asdict(config.run.split),
        },
        "missing_policy": config.run.missing_policy.value,
        "training_seed": config.run.training_seed,
        "sample_seed": config.run.sample_seed,
        "device": config.run.device,
        "sampler_seed": config.sampler_seed,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _class_counts_payload(
    counts: tuple[ClassCount, ...] | None,
) -> list[dict[str, object]] | None:
    if counts is None:
        return None
    return [
        {
            "label_repr": repr(item.label),
            "label_type": type(item.label).__name__,
            "count": item.count,
        }
        for item in counts
    ]


def _missing_report_payload(report: MissingReport) -> dict[str, object]:
    return {
        "policy": report.policy.value,
        "rows_before": report.rows_before,
        "rows_after": report.rows_after,
        "dropped_count": report.dropped_count,
        "dropped_fraction": report.dropped_fraction,
        "missing_by_column": dict(report.missing_by_column),
        "class_counts_before": _class_counts_payload(
            report.class_counts_before
        ),
        "class_counts_after": _class_counts_payload(report.class_counts_after),
    }


def _record_score_evidence(
    trial: optuna.Trial,
    score: TuningScore,
    *,
    fit_seconds: float,
    sample_seconds: float,
    missing_report: MissingReport,
) -> None:
    trial.set_user_attr(
        "mean_standardized_wasserstein",
        score.mean_wasserstein,
    )
    trial.set_user_attr(
        "mean_jensen_shannon",
        score.mean_jensen_shannon,
    )
    trial.set_user_attr(
        "column_scores",
        {
            item.column: {
                "kind": item.kind.value,
                "metric": item.metric.value,
                "value": item.value,
                "reference_scale": item.reference_scale,
            }
            for item in score.columns
        },
    )
    trial.set_user_attr("fit_seconds", fit_seconds)
    trial.set_user_attr("sample_seconds", sample_seconds)
    trial.set_user_attr(
        "missing_report",
        _missing_report_payload(missing_report),
    )


def _completed_trials(study: optuna.Study) -> tuple[optuna.trial.FrozenTrial, ...]:
    return tuple(
        trial
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE
    )


def _initialize_or_validate_study(
    study: optuna.Study,
    *,
    fingerprint: str,
    config: OptunaStudyConfig,
) -> None:
    stored = study.user_attrs.get(_FINGERPRINT_ATTR)
    if stored is None:
        if study.trials:
            raise ContractViolation(
                "Cannot resume an unversioned benchmark tuning study; start a "
                "new study or use an explicitly reviewed migration."
            )
        study.set_user_attr(_FINGERPRINT_ATTR, fingerprint)
        study.set_user_attr("harness_version", OPTUNA_HARNESS_VERSION)
        study.set_user_attr("protocol_version", config.protocol_version)
        study.set_user_attr("objective_version", config.objective_version)
    elif stored != fingerprint:
        raise ContractViolation(
            "Refusing to mix tuning trials from a different dataset, semantic "
            "view, holdout, seed, device, sampler, model, objective, or "
            "search/failure protocol."
        )

    running = tuple(
        trial.number
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.RUNNING
    )
    if running:
        if not config.load_if_exists:
            raise ContractViolation(
                "A non-resumed study unexpectedly contains running trials."
            )
        for trial_number in running:
            study.tell(trial_number, state=optuna.trial.TrialState.FAIL)
        study.set_user_attr("recovered_running_trials", list(running))


def _record_failure(
    trial: optuna.Trial,
    error: BaseException,
    failure: TrialFailure | None,
) -> TrialFailureAction:
    if failure is None and isinstance(error, optuna.TrialPruned):
        failure = TrialFailure(
            action=TrialFailureAction.PRUNE,
            kind="optuna_trial_pruned",
            message=str(error) or "Trial pruned by model-owned search logic.",
        )
    if failure is None:
        failure = TrialFailure(
            action=TrialFailureAction.RAISE,
            kind=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error) or repr(error),
        )
    if not isinstance(failure, TrialFailure):
        raise ContractViolation(
            "classify_failure must return TrialFailure or None."
        )
    trial.set_user_attr(
        "failure",
        {
            "action": failure.action.value,
            "kind": failure.kind,
            "message": failure.message,
        },
    )
    return failure.action


def run_optuna_holdout_study(
    dataset: TabularDataset,
    *,
    model_name: str,
    input_spec: InputSpec,
    config: OptunaStudyConfig,
    suggest_config: Callable[[optuna.Trial], NativeConfigT],
    make_adapter: Callable[[NativeConfigT], ModelAdapter],
    encode_config: Callable[[NativeConfigT], Mapping[str, object]],
    decode_config: Callable[[object], NativeConfigT],
    classify_failure: (
        Callable[[BaseException], TrialFailure | None] | None
    ) = None,
) -> OptunaTuningResult[NativeConfigT]:
    """Run or resume one common holdout study until its success target.

    Each trial uses the same raw holdout membership, native training/sample
    seeds, fold-local codec behavior, and decoded raw-space tuning score. The
    model callbacks own only native configuration and error semantics.

    The persistent study remains usable when this function raises because of a
    timeout, attempt ceiling, incompatible resume, or propagated native error.
    A result is returned only after ``target_complete_trials`` successful
    objective values exist.
    """

    if not isinstance(dataset, TabularDataset):
        raise ContractViolation("dataset must be TabularDataset.")
    validate_tabular_dataset(dataset)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ContractViolation("model_name must be a non-empty string.")
    validate_input_spec(input_spec)
    if not isinstance(config, OptunaStudyConfig):
        raise ContractViolation("config must be OptunaStudyConfig.")
    for callback_name, callback in (
        ("suggest_config", suggest_config),
        ("make_adapter", make_adapter),
        ("encode_config", encode_config),
        ("decode_config", decode_config),
    ):
        if not callable(callback):
            raise ContractViolation(f"{callback_name} must be callable.")
    if classify_failure is not None and not callable(classify_failure):
        raise ContractViolation("classify_failure must be callable or None.")

    fingerprint = _study_fingerprint(
        dataset,
        model_name,
        input_spec,
        config,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.sampler_seed),
        pruner=optuna.pruners.NopPruner(),
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=config.load_if_exists,
    )
    _initialize_or_validate_study(
        study,
        fingerprint=fingerprint,
        config=config,
    )

    started = monotonic()
    while len(_completed_trials(study)) < config.target_complete_trials:
        if len(study.trials) >= config.max_total_trials:
            break
        if config.timeout_seconds is not None and (
            monotonic() - started >= config.timeout_seconds
        ):
            break

        trial = study.ask()
        try:
            native_config = suggest_config(trial)
            native_payload = _json_mapping(
                encode_config(native_config),
                label="encoded native config",
            )
            trial.set_user_attr(_NATIVE_CONFIG_ATTR, native_payload)
            adapter = make_adapter(native_config)
            validate_adapter_definition(adapter)
            if adapter.name != model_name:
                raise ContractViolation(
                    "make_adapter returned a different model name: "
                    f"{adapter.name!r}, expected {model_name!r}."
                )
            if adapter.input_spec != input_spec:
                raise ContractViolation(
                    "make_adapter returned InputSpec different from the "
                    "fingerprinted study InputSpec."
                )
            trial_run = replace(
                config.run,
                run_id=f"{config.run.run_id}-trial-{trial.number}",
                artifact_dir=config.run.artifact_dir / f"trial-{trial.number}",
            )
            holdout = run_holdout_trial(
                dataset,
                lambda adapter=adapter: adapter,
                trial_run,
            )
            score = evaluate_tuning_score(
                holdout.dataset,
                holdout.train_raw,
                holdout.validation_raw,
                holdout.synthetic_raw,
            )
            _record_score_evidence(
                trial,
                score,
                fit_seconds=holdout.fit_seconds,
                sample_seconds=holdout.sample_seconds,
                missing_report=holdout.missing_report,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit, CancelledError):
            # Process-control signals deliberately leave the trial RUNNING.
            # A persistent study recovers it as FAIL on the next invocation.
            raise
        except BaseException as error:
            try:
                failure = (
                    classify_failure(error)
                    if classify_failure is not None
                    else None
                )
                action = _record_failure(trial, error, failure)
            except (KeyboardInterrupt, SystemExit, GeneratorExit, CancelledError):
                raise
            except BaseException:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                raise
            if action is TrialFailureAction.PRUNE:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            if action is TrialFailureAction.FAIL:
                continue
            raise
        else:
            study.tell(trial, score.total)

    complete = _completed_trials(study)
    if len(complete) < config.target_complete_trials:
        pruned = sum(
            trial.state is optuna.trial.TrialState.PRUNED
            for trial in study.trials
        )
        failed = sum(
            trial.state is optuna.trial.TrialState.FAIL
            for trial in study.trials
        )
        raise ContractViolation(
            "Tuning stopped before its successful-trial target: "
            f"complete={len(complete)}, "
            f"target={config.target_complete_trials}, pruned={pruned}, "
            f"failed={failed}, total={len(study.trials)}. Resume the same "
            "compatible study after reviewing failures or increasing the "
            "operational budget."
        )

    best_trial = study.best_trial
    best_payload = _json_mapping(
        best_trial.user_attrs.get(_NATIVE_CONFIG_ATTR),
        label="best trial native config",
    )
    try:
        best_config = decode_config(best_payload)
    except ContractViolation:
        raise
    except Exception as error:
        raise ContractViolation(
            "decode_config could not reconstruct the best native config."
        ) from error
    round_trip = _json_mapping(
        encode_config(best_config),
        label="re-encoded best native config",
    )
    if round_trip != best_payload:
        raise ContractViolation(
            "Native config encode/decode callbacks are not reversible for the "
            "best trial."
        )
    if best_trial.value is None or not np.isfinite(float(best_trial.value)):
        raise ContractViolation("Best completed tuning score is not finite.")
    return OptunaTuningResult(
        study=study,
        config=config,
        dataset=dataset,
        model_name=model_name,
        input_spec=input_spec,
        best_config=best_config,
        best_config_payload=best_payload,
        best_score=float(best_trial.value),
        fingerprint=fingerprint,
    )


def _run_payload(config: HoldoutRunConfig) -> dict[str, object]:
    return {
        "split": {
            "type": type(config.split).__name__,
            **asdict(config.split),
        },
        "missing_policy": config.missing_policy.value,
        "run_id": config.run_id,
        "training_seed": config.training_seed,
        "sample_seed": config.sample_seed,
        "device": config.device,
        "artifact_dir": str(config.artifact_dir),
    }


def write_optuna_tuning_artifacts(
    result: OptunaTuningResult[object],
    output_dir: Path,
) -> Path:
    """Create portable review evidence for one completed common study.

    The destination is create-only and committed atomically. It stores the
    reconstructed best native config, every Optuna trial, and one manifest.
    The Optuna storage URI is deliberately omitted because it may contain
    credentials. A model-specific later selection phase may write its own
    separate linked artifact instead of extending this directory in place.
    """

    if not isinstance(result, OptunaTuningResult):
        raise ContractViolation("result must be OptunaTuningResult.")
    if not isinstance(output_dir, Path):
        raise ContractViolation("output_dir must be pathlib.Path.")
    if output_dir.exists():
        raise ContractViolation(
            f"Tuning artifact directory already exists: {output_dir}."
        )

    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in result.study.trials
    ]
    complete = sum(
        trial.state is optuna.trial.TrialState.COMPLETE
        for trial in result.study.trials
    )
    pruned = sum(
        trial.state is optuna.trial.TrialState.PRUNED
        for trial in result.study.trials
    )
    failed = sum(
        trial.state is optuna.trial.TrialState.FAIL
        for trial in result.study.trials
    )
    manifest = {
        "artifact_type": "benchmark_optuna_tuning",
        "artifact_version": OPTUNA_TUNING_ARTIFACT_VERSION,
        "harness_version": OPTUNA_HARNESS_VERSION,
        "model_name": result.model_name,
        "dataset": {
            "name": result.dataset.name,
            "target": result.dataset.target,
            "task": (
                result.dataset.task.value
                if result.dataset.task is not None
                else None
            ),
            "columns": [
                {"name": column.name, "kind": column.kind.value}
                for column in result.dataset.columns
            ],
        },
        "input_spec": {
            "continuous_view": result.input_spec.continuous_view.value,
            "discrete_view": result.input_spec.discrete_view.value,
            "categorical_view": result.input_spec.categorical_view.value,
        },
        "fingerprint": result.fingerprint,
        "study": {
            "name": result.study.study_name,
            "direction": result.study.direction.name,
            "sampler": type(result.study.sampler).__name__,
            "sampler_seed": result.config.sampler_seed,
            "protocol_version": result.config.protocol_version,
            "objective_version": result.config.objective_version,
            "target_complete_trials": result.config.target_complete_trials,
            "max_total_trials": result.config.max_total_trials,
            "completed_trials": complete,
            "pruned_trials": pruned,
            "failed_trials": failed,
            "timeout_seconds": result.config.timeout_seconds,
            "storage_configured": result.config.storage is not None,
            "load_if_exists": result.config.load_if_exists,
        },
        "holdout_run": _run_payload(result.config.run),
        "missing_report": result.study.best_trial.user_attrs.get(
            "missing_report"
        ),
        "best_trial": result.study.best_trial.number,
        "best_score": result.best_score,
        "files": {
            "best_config": "best-config.json",
            "trials": "trials.json",
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(
        f".{output_dir.name}.initializing-{uuid4().hex}"
    )
    temporary.mkdir()
    try:
        (temporary / "best-config.json").write_text(
            json.dumps(
                dict(result.best_config_payload),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "trials.json").write_text(
            json.dumps(
                trials,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except FileExistsError as error:
        raise ContractViolation(
            f"Tuning artifact directory already exists: {output_dir}."
        ) from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / "manifest.json"

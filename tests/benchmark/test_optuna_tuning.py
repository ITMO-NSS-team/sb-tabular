"""Tests for the reusable model-owned Optuna lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import optuna
import pandas as pd

from sbtab.benchmark import (
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    ContractViolation,
    DiscreteView,
    HoldoutConfig,
    HoldoutRunConfig,
    InputSpec,
    MissingPolicy,
    PreparedTable,
    RunContext,
    TabularDataset,
)
from sbtab.benchmark.tuning import (
    OptunaStudyConfig,
    TrialFailure,
    TrialFailureAction,
    run_optuna_holdout_study,
    write_optuna_tuning_artifacts,
)


@dataclass(frozen=True)
class _NativeConfig:
    shift: float
    fail_code: int = -1


class _NumericalError(RuntimeError):
    def __init__(self, code: int) -> None:
        super().__init__(f"numerical failure {code}")
        self.code = code


class _NumericalSignal(BaseException):
    """Mirror a native numerical failure with a legacy BaseException base."""


class _TunableAdapter:
    name = "tunable-echo"
    input_spec = InputSpec(
        continuous_view=ContinuousView.STANDARD,
        discrete_view=DiscreteView.FINITE_STATE_CODES,
        categorical_view=CategoricalView.FINITE_STATE_CODES,
    )

    def __init__(self, config: _NativeConfig) -> None:
        self.config = config

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        if self.config.fail_code >= 0:
            raise _NumericalError(self.config.fail_code)
        self.train = train

    def sample(self, n: int, seed: int) -> PreparedTable:
        frame = self.train.frame.iloc[:n].copy()
        frame["value"] += self.config.shift
        return PreparedTable(frame=frame, schema=self.train.schema)


def _dataset(*, first_value: float = 0.0) -> TabularDataset:
    return TabularDataset(
        name="optuna-dataset",
        frame=pd.DataFrame(
            {
                "value": [first_value] + [float(index) for index in range(1, 10)],
                "group": ["a", "b"] * 5,
            }
        ),
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("group", ColumnKind.CATEGORICAL),
        ),
    )


def _run_config() -> HoldoutRunConfig:
    return HoldoutRunConfig(
        split=HoldoutConfig(validation_fraction=0.2, seed=5),
        missing_policy=MissingPolicy.COMPLETE_CASE,
        run_id="optuna-test",
        training_seed=11,
        sample_seed=22,
        artifact_dir=Path("logical-optuna-artifacts"),
    )


def _study_config(
    *,
    target: int = 2,
    maximum: int = 3,
    storage: str | None = None,
    load_if_exists: bool = False,
) -> OptunaStudyConfig:
    return OptunaStudyConfig(
        run=_run_config(),
        target_complete_trials=target,
        max_total_trials=maximum,
        sampler_seed=7,
        study_name="optuna-harness-test",
        storage=storage,
        load_if_exists=load_if_exists,
        protocol_version=3,
        objective_version=2,
    )


def _suggest_config(trial: optuna.Trial) -> _NativeConfig:
    return _NativeConfig(
        shift=trial.suggest_float("shift", 0.0, 1.0),
    )


def _encode_config(config: _NativeConfig) -> dict[str, object]:
    return {"shift": config.shift, "fail_code": config.fail_code}


def _decode_config(payload: object) -> _NativeConfig:
    if not isinstance(payload, dict):
        raise ContractViolation("native payload must be dict")
    return _NativeConfig(
        shift=float(payload["shift"]),
        fail_code=int(payload["fail_code"]),
    )


def _run_study(
    config: OptunaStudyConfig,
    *,
    dataset: TabularDataset | None = None,
    suggest_config=_suggest_config,
    classify_failure=None,
):
    return run_optuna_holdout_study(
        _dataset() if dataset is None else dataset,
        model_name=_TunableAdapter.name,
        input_spec=_TunableAdapter.input_spec,
        config=config,
        suggest_config=suggest_config,
        make_adapter=_TunableAdapter,
        encode_config=_encode_config,
        decode_config=_decode_config,
        classify_failure=classify_failure,
    )


class OptunaHarnessTests(unittest.TestCase):
    """Verify reusable lifecycle behavior without a concrete model family."""

    def test_common_holdout_objective_records_evidence_and_best_config(self) -> None:
        result = _run_study(_study_config())

        self.assertEqual(len(result.study.trials), 2)
        self.assertEqual(
            [trial.state for trial in result.study.trials],
            [
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.COMPLETE,
            ],
        )
        self.assertEqual(result.best_config_payload, _encode_config(result.best_config))
        self.assertEqual(result.best_score, result.study.best_value)
        self.assertEqual(len(result.fingerprint), 64)
        evidence = result.study.best_trial.user_attrs
        self.assertIn("mean_standardized_wasserstein", evidence)
        self.assertIn("mean_jensen_shannon", evidence)
        self.assertEqual(tuple(evidence["column_scores"]), ("value", "group"))
        self.assertEqual(evidence["missing_report"]["rows_after"], 10)
        self.assertGreaterEqual(evidence["fit_seconds"], 0.0)
        self.assertGreaterEqual(evidence["sample_seconds"], 0.0)

    def test_resume_adds_only_missing_successes_and_rejects_changed_data(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            storage = f"sqlite:///{Path(temporary_dir) / 'study.sqlite3'}"
            first = _run_study(
                _study_config(
                    target=1,
                    maximum=2,
                    storage=storage,
                )
            )
            self.assertEqual(len(first.study.trials), 1)

            resumed = _run_study(
                _study_config(
                    target=2,
                    maximum=3,
                    storage=storage,
                    load_if_exists=True,
                )
            )
            self.assertEqual(len(resumed.study.trials), 2)
            self.assertEqual(
                sum(
                    trial.state is optuna.trial.TrialState.COMPLETE
                    for trial in resumed.study.trials
                ),
                2,
            )

            adapter_calls = 0

            def forbidden_adapter(config: _NativeConfig) -> _TunableAdapter:
                nonlocal adapter_calls
                adapter_calls += 1
                return _TunableAdapter(config)

            with self.assertRaisesRegex(ContractViolation, "Refusing to mix"):
                run_optuna_holdout_study(
                    _dataset(first_value=999.0),
                    model_name=_TunableAdapter.name,
                    input_spec=_TunableAdapter.input_spec,
                    config=_study_config(
                        target=2,
                        maximum=3,
                        storage=storage,
                        load_if_exists=True,
                    ),
                    suggest_config=_suggest_config,
                    make_adapter=forbidden_adapter,
                    encode_config=_encode_config,
                    decode_config=_decode_config,
                )
            self.assertEqual(adapter_calls, 0)

    def test_pruned_and_failed_attempts_continue_to_success_target(self) -> None:
        def suggest(trial: optuna.Trial) -> _NativeConfig:
            trial.suggest_int("fixed", 0, 0)
            if trial.number == 0:
                raise optuna.TrialPruned("known infeasible region")
            return _NativeConfig(shift=0.0, fail_code=trial.number)

        def classify(error: Exception) -> TrialFailure | None:
            if not isinstance(error, _NumericalError):
                return None
            return TrialFailure(
                action=TrialFailureAction.FAIL,
                kind=f"numerical_{error.code}",
                message=str(error),
            )

        def eventually_succeeds(trial: optuna.Trial) -> _NativeConfig:
            config = suggest(trial)
            return (
                _NativeConfig(shift=0.0)
                if trial.number == 2
                else config
            )

        result = _run_study(
            _study_config(target=1, maximum=3),
            suggest_config=eventually_succeeds,
            classify_failure=classify,
        )

        self.assertEqual(
            [trial.state for trial in result.study.trials],
            [
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.COMPLETE,
            ],
        )
        self.assertEqual(
            result.study.trials[0].user_attrs["failure"]["action"],
            "prune",
        )
        self.assertEqual(
            result.study.trials[1].user_attrs["failure"]["action"],
            "fail",
        )

    def test_unclassified_exception_is_failed_and_propagated(self) -> None:
        def broken(_trial: optuna.Trial) -> _NativeConfig:
            raise RuntimeError("unexpected model error")

        with TemporaryDirectory() as temporary_dir:
            storage = f"sqlite:///{Path(temporary_dir) / 'study.sqlite3'}"
            with self.assertRaisesRegex(RuntimeError, "unexpected model error"):
                _run_study(
                    _study_config(
                        target=1,
                        maximum=1,
                        storage=storage,
                    ),
                    suggest_config=broken,
                )
            study = optuna.load_study(
                study_name="optuna-harness-test",
                storage=storage,
            )
            self.assertEqual(
                study.trials[0].state,
                optuna.trial.TrialState.FAIL,
            )
            self.assertEqual(
                study.trials[0].user_attrs["failure"]["action"],
                "raise",
            )

    def test_model_can_classify_legacy_base_exception(self) -> None:
        def suggest(trial: optuna.Trial) -> _NativeConfig:
            if trial.number == 0:
                raise _NumericalSignal("non-finite native loss")
            return _NativeConfig(shift=0.0)

        def classify(error: BaseException) -> TrialFailure | None:
            if isinstance(error, _NumericalSignal):
                return TrialFailure(
                    action=TrialFailureAction.FAIL,
                    kind="non_finite_native_loss",
                    message=str(error),
                )
            return None

        result = _run_study(
            _study_config(target=1, maximum=2),
            suggest_config=suggest,
            classify_failure=classify,
        )

        self.assertEqual(
            [trial.state for trial in result.study.trials],
            [
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.COMPLETE,
            ],
        )

    def test_running_trial_is_failed_before_resume(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            storage = f"sqlite:///{Path(temporary_dir) / 'study.sqlite3'}"
            first = _run_study(
                _study_config(target=1, maximum=3, storage=storage)
            )
            abandoned = first.study.ask()
            self.assertEqual(abandoned.number, 1)

            resumed = _run_study(
                _study_config(
                    target=2,
                    maximum=4,
                    storage=storage,
                    load_if_exists=True,
                )
            )
            self.assertEqual(
                resumed.study.trials[1].state,
                optuna.trial.TrialState.FAIL,
            )
            self.assertEqual(
                resumed.study.user_attrs["recovered_running_trials"],
                [1],
            )

    def test_attempt_ceiling_has_no_partial_result(self) -> None:
        def always_fails(trial: optuna.Trial) -> _NativeConfig:
            return _NativeConfig(shift=0.0, fail_code=trial.number)

        def classify(error: Exception) -> TrialFailure | None:
            if isinstance(error, _NumericalError):
                return TrialFailure(
                    action=TrialFailureAction.FAIL,
                    kind="expected_numerical_failure",
                    message=str(error),
                )
            return None

        with self.assertRaisesRegex(
            ContractViolation,
            "complete=0, target=1, pruned=0, failed=2, total=2",
        ):
            _run_study(
                _study_config(target=1, maximum=2),
                suggest_config=always_fails,
                classify_failure=classify,
            )

    def test_artifacts_are_create_only_and_omit_storage_uri(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            storage = f"sqlite:///{Path(temporary_dir) / 'study.sqlite3'}"
            result = _run_study(
                _study_config(
                    target=1,
                    maximum=1,
                    storage=storage,
                )
            )
            output_dir = Path(temporary_dir) / "tuning"
            manifest_path = write_optuna_tuning_artifacts(result, output_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(
                manifest["artifact_type"],
                "benchmark_optuna_tuning",
            )
            self.assertEqual(manifest["model_name"], _TunableAdapter.name)
            self.assertEqual(manifest["study"]["completed_trials"], 1)
            self.assertNotIn("storage", manifest["study"])
            self.assertTrue(manifest["study"]["storage_configured"])
            self.assertNotIn(storage, json.dumps(manifest))
            self.assertTrue((output_dir / "best-config.json").is_file())
            self.assertTrue((output_dir / "trials.json").is_file())

            with self.assertRaisesRegex(ContractViolation, "already exists"):
                write_optuna_tuning_artifacts(result, output_dir)

    def test_config_rejects_ambiguous_operational_values(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "cannot be below"):
            _study_config(target=3, maximum=2)
        with self.assertRaisesRegex(ContractViolation, "sampler_seed"):
            OptunaStudyConfig(run=_run_config(), sampler_seed=True)
        with self.assertRaisesRegex(ContractViolation, "timeout_seconds"):
            OptunaStudyConfig(run=_run_config(), timeout_seconds=float("inf"))


if __name__ == "__main__":
    unittest.main()

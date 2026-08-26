"""Tests for crash-safe fold checkpoints and linked evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from sbtab.benchmark import (
    BenchmarkConfig,
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    ContractViolation,
    DiscreteView,
    InputSpec,
    KFoldConfig,
    MissingPolicy,
    PreparedTable,
    RunContext,
    TabularDataset,
    run_cross_validation_resumable,
)
from sbtab.evaluation import (
    evaluate_cross_validation,
    write_evaluation_artifacts,
)


class _EchoAdapter:
    name = "artifact-echo"
    input_spec = InputSpec(
        continuous_view=ContinuousView.RAW,
        discrete_view=DiscreteView.FINITE_STATE_CODES,
        categorical_view=CategoricalView.FINITE_STATE_CODES,
    )

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        self.train = train

    def sample(self, n: int, seed: int) -> PreparedTable:
        return PreparedTable(
            frame=self.train.frame.iloc[:n].copy(),
            schema=self.train.schema,
        )


def _dataset() -> TabularDataset:
    return TabularDataset(
        name="artifact-dataset",
        frame=pd.DataFrame(
            {
                "row_id": [
                    None if index == 0 else f"id-{index}"
                    for index in range(8)
                ],
                "value": [float(index) for index in range(8)],
                "group": ["a", "b"] * 4,
            }
        ),
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("group", ColumnKind.CATEGORICAL),
        ),
        identifier="row_id",
    )


def _config(*, seed: int = 42) -> BenchmarkConfig:
    return BenchmarkConfig(
        split=KFoldConfig(n_splits=2, seed=seed),
        missing_policy=MissingPolicy.COMPLETE_CASE,
        run_id="artifact-test",
        artifact_dir=Path("logical-artifact-root"),
    )


class CrossValidationArtifactTests(unittest.TestCase):
    """Verify immediate fold commits, safe resume, and immutable linkage."""

    def test_interrupted_run_resumes_only_missing_folds(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "generation"
            calls = 0

            def interrupted_factory() -> _EchoAdapter:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return _EchoAdapter()

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_cross_validation_resumable(
                    _dataset(),
                    interrupted_factory,
                    _config(),
                    output_dir=output_dir,
                )

            self.assertTrue((output_dir / "fold-0" / "fold.json").is_file())
            self.assertFalse((output_dir / "fold-1").exists())
            self.assertFalse((output_dir / "manifest.json").exists())

            resumed_calls = 0

            def resumed_factory() -> _EchoAdapter:
                nonlocal resumed_calls
                resumed_calls += 1
                return _EchoAdapter()

            completed = run_cross_validation_resumable(
                _dataset(),
                resumed_factory,
                _config(),
                output_dir=output_dir,
            )

            self.assertEqual(resumed_calls, 1)
            self.assertEqual(completed.resumed_fold_ids, (0,))
            self.assertEqual(completed.generated_fold_ids, (1,))
            self.assertTrue(completed.manifest_path.is_file())
            self.assertEqual(len(completed.result.folds), 2)

            def forbidden_factory() -> _EchoAdapter:
                raise AssertionError("completed folds must not be retrained")

            repeated = run_cross_validation_resumable(
                _dataset(),
                forbidden_factory,
                _config(),
                output_dir=output_dir,
            )
            self.assertEqual(repeated.resumed_fold_ids, (0, 1))
            self.assertEqual(repeated.generated_fold_ids, ())

    def test_resume_rejects_changed_plan_and_corrupt_fold(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "generation"
            completed = run_cross_validation_resumable(
                _dataset(),
                _EchoAdapter,
                _config(),
                output_dir=output_dir,
            )

            with self.assertRaisesRegex(ContractViolation, "does not match"):
                run_cross_validation_resumable(
                    _dataset(),
                    _EchoAdapter,
                    _config(seed=7),
                    output_dir=output_dir,
                )

            changed = _dataset()
            changed_frame = changed.frame.copy()
            changed_frame.loc[0, "value"] = 999.0
            changed = TabularDataset(
                name=changed.name,
                frame=changed_frame,
                columns=changed.columns,
                identifier=changed.identifier,
            )
            with self.assertRaisesRegex(ContractViolation, "dataset values"):
                run_cross_validation_resumable(
                    changed,
                    _EchoAdapter,
                    _config(),
                    output_dir=output_dir,
                )

            manifest = json.loads(
                completed.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["artifact_version"], 2)
            synthetic_path = output_dir / "fold-0" / "synthetic.json"
            synthetic_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "checksum"):
                run_cross_validation_resumable(
                    _dataset(),
                    _EchoAdapter,
                    _config(),
                    output_dir=output_dir,
                )

    def test_evaluation_manifest_links_exact_generation_bytes(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            generation = run_cross_validation_resumable(
                _dataset(),
                _EchoAdapter,
                _config(),
                output_dir=root / "generation",
            )
            evaluation = evaluate_cross_validation(generation.result)

            evaluation_manifest = write_evaluation_artifacts(
                evaluation,
                root / "evaluation",
                generation_manifest=generation.manifest_path,
            )

            payload = json.loads(
                evaluation_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["artifact_type"],
                "cross_validation_evaluation",
            )
            self.assertEqual(payload["artifact_version"], 2)
            self.assertEqual(len(payload["generation_manifest_sha256"]), 64)

            with self.assertRaisesRegex(ContractViolation, "already exists"):
                write_evaluation_artifacts(
                    evaluation,
                    root / "evaluation",
                    generation_manifest=generation.manifest_path,
                )

            generation.result.folds[0].synthetic_raw.loc[0, "value"] = 999.0
            with self.assertRaisesRegex(ContractViolation, "content"):
                write_evaluation_artifacts(
                    evaluation,
                    root / "mismatched-evaluation",
                    generation_manifest=generation.manifest_path,
                )


if __name__ == "__main__":
    unittest.main()

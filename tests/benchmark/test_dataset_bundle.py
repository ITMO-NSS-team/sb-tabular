"""Tests for portable mixed-dataset acquisition bundles."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from sbtab.benchmark import ColumnKind, ContractViolation
from sbtab.benchmark.datasets import (
    MIXED_DATASET_COLUMNS,
    load_mixed_dataset_bundle,
    make_mixed_dataset,
    write_mixed_dataset_bundle,
)


def _frame(key: str) -> pd.DataFrame:
    values: dict[str, list[object]] = {}
    for column in MIXED_DATASET_COLUMNS[key]:
        if column.kind is ColumnKind.CONTINUOUS:
            values[column.name] = [1.5, 2.5]
        elif column.kind is ColumnKind.DISCRETE:
            values[column.name] = [1, 2]
        elif column.ordered_values is not None:
            values[column.name] = [
                column.ordered_values[0],
                column.ordered_values[-1],
            ]
        else:
            values[column.name] = ["first", "second"]
    return pd.DataFrame(values)


class DatasetBundleTests(unittest.TestCase):
    """Verify cross-version table format, selection, and corruption checks."""

    def test_bundle_round_trip_preserves_order_and_semantic_values(self) -> None:
        datasets = tuple(
            make_mixed_dataset(key, _frame(key))
            for key in ("adult", "diamonds")
        )
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "mixed-v1"
            manifest_path = write_mixed_dataset_bundle(datasets, output_dir)

            loaded = load_mixed_dataset_bundle(
                output_dir,
                keys=("diamonds", "adult"),
            )

            self.assertEqual(tuple(dataset.name for dataset in loaded), ("diamonds", "adult"))
            pd.testing.assert_frame_equal(loaded[0].frame, datasets[1].frame)
            pd.testing.assert_frame_equal(loaded[1].frame, datasets[0].frame)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_type"], "mixed_dataset_bundle")
            self.assertEqual(manifest["artifact_version"], 1)
            self.assertEqual(len(manifest["datasets"][0]["table_sha256"]), 64)

    def test_bundle_rejects_overwrite_missing_key_and_corrupt_table(self) -> None:
        dataset = make_mixed_dataset("adult", _frame("adult"))
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "mixed-v1"
            write_mixed_dataset_bundle((dataset,), output_dir)

            with self.assertRaisesRegex(ContractViolation, "already exists"):
                write_mixed_dataset_bundle((dataset,), output_dir)
            with self.assertRaisesRegex(ContractViolation, "lacks requested"):
                load_mixed_dataset_bundle(output_dir, keys=("diamonds",))

            (output_dir / "adult.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "checksum"):
                load_mixed_dataset_bundle(output_dir)


if __name__ == "__main__":
    unittest.main()

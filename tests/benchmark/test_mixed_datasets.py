"""Contract tests for the fourteen explicit mixed-dataset declarations."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from sbtab.benchmark import ColumnKind, ContractViolation, TaskType
from sbtab.benchmark.datasets import (
    MIXED_DATASET_COLUMNS,
    MIXED_DATASET_KEYS,
    MIXED_DATASET_SOURCES,
    fetch_mixed_dataset,
    make_mixed_dataset,
)


_EXPECTED_TARGETS = {
    "adult": ("income", TaskType.CLASSIFICATION),
    "credit_approval": ("A16", TaskType.CLASSIFICATION),
    "online_shoppers": ("Revenue", TaskType.CLASSIFICATION),
    "eucalyptus": ("Utility", TaskType.CLASSIFICATION),
    "forest_fires": ("area", TaskType.REGRESSION),
    "insurance": ("charges", TaskType.REGRESSION),
    "house_sales": ("price", TaskType.REGRESSION),
    "cardiovascular_disease": ("cardio", TaskType.CLASSIFICATION),
    "churn_modelling": ("Exited", TaskType.CLASSIFICATION),
    "auto_mpg": ("mpg", TaskType.REGRESSION),
    "diamonds": ("price", TaskType.REGRESSION),
    "real_estate": (
        "Y house price of unit area",
        TaskType.REGRESSION,
    ),
    "stroke_prediction": ("stroke", TaskType.CLASSIFICATION),
    "palmer_penguins": ("Species", TaskType.CLASSIFICATION),
}


def _one_row_frame(key: str) -> pd.DataFrame:
    """Construct values that satisfy only the declared semantic dtypes."""

    values: dict[str, list[object]] = {}
    for column in MIXED_DATASET_COLUMNS[key]:
        if column.kind is ColumnKind.CONTINUOUS:
            values[column.name] = [1.5]
        elif column.kind is ColumnKind.DISCRETE:
            values[column.name] = [1]
        else:
            values[column.name] = [
                column.ordered_values[0]
                if column.ordered_values is not None
                else "state"
            ]
    return pd.DataFrame(values)


class MixedDatasetDeclarationTests(unittest.TestCase):
    """Verify all published schemas without downloading external data."""

    def test_collection_contains_exactly_fourteen_unique_keys(self) -> None:
        self.assertEqual(len(MIXED_DATASET_KEYS), 14)
        self.assertEqual(len(set(MIXED_DATASET_KEYS)), 14)
        self.assertEqual(tuple(MIXED_DATASET_COLUMNS), MIXED_DATASET_KEYS)
        self.assertEqual(tuple(MIXED_DATASET_SOURCES), MIXED_DATASET_KEYS)
        self.assertEqual(set(_EXPECTED_TARGETS), set(MIXED_DATASET_KEYS))

    def test_every_source_names_one_exact_table(self) -> None:
        for key, source in MIXED_DATASET_SOURCES.items():
            with self.subTest(dataset=key):
                self.assertIn(source.provider, {"uci", "openml", "kaggle"})
                self.assertTrue(source.locator)
                self.assertTrue(source.table)

    def test_every_declaration_builds_and_models_its_target(self) -> None:
        for key in MIXED_DATASET_KEYS:
            with self.subTest(dataset=key):
                dataset = make_mixed_dataset(key, _one_row_frame(key))
                target, task = _EXPECTED_TARGETS[key]

                self.assertEqual(dataset.column_order, tuple(dataset.frame.columns))
                self.assertEqual(dataset.target, target)
                self.assertEqual(dataset.task, task)
                self.assertIn(target, dataset.column_order)
                expected_target_kind = (
                    ColumnKind.CATEGORICAL
                    if task is TaskType.CLASSIFICATION
                    else ColumnKind.CONTINUOUS
                )
                self.assertEqual(dataset.column(target).kind, expected_target_kind)

    def test_online_shoppers_reuses_the_approved_pilot_declaration(self) -> None:
        dataset = make_mixed_dataset(
            "online_shoppers",
            _one_row_frame("online_shoppers"),
        )

        self.assertEqual(
            dataset.discrete_columns,
            ("Administrative", "Informational", "ProductRelated"),
        )
        self.assertIn("TrafficType", dataset.categorical_columns)
        self.assertIn("Revenue", dataset.categorical_columns)

    def test_semantic_review_does_not_treat_codes_as_magnitudes(self) -> None:
        expected_categorical = {
            "house_sales": {"waterfront", "zipcode"},
            "cardiovascular_disease": {"gender", "smoke", "alco", "active"},
            "churn_modelling": {"HasCrCard", "IsActiveMember"},
            "auto_mpg": {"origin"},
            "stroke_prediction": {"hypertension", "heart_disease"},
        }
        for key, names in expected_categorical.items():
            with self.subTest(dataset=key):
                declaration = {column.name: column for column in MIXED_DATASET_COLUMNS[key]}
                self.assertTrue(
                    all(declaration[name].kind is ColumnKind.CATEGORICAL for name in names)
                )

        diamonds = {column.name: column for column in MIXED_DATASET_COLUMNS["diamonds"]}
        for name in ("cut", "color", "clarity"):
            self.assertIsNotNone(diamonds[name].ordered_values)

    def test_unknown_key_fails_before_frame_validation(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown"):
            make_mixed_dataset("unknown", pd.DataFrame())

    def test_extra_source_column_is_not_silently_dropped(self) -> None:
        frame = _one_row_frame("adult").assign(source_id=[42])

        with self.assertRaisesRegex(ContractViolation, "source_id"):
            make_mixed_dataset("adult", frame)

    def test_fetch_boundary_returns_a_validated_dataset(self) -> None:
        frame = _one_row_frame("adult")
        fake_fetchers = {"adult": lambda: frame}

        with patch(
            "sbtab.benchmark.datasets.acquisition._FETCHER_BY_KEY",
            fake_fetchers,
        ):
            dataset = fetch_mixed_dataset("adult")

        self.assertEqual(dataset.name, "adult")
        self.assertEqual(dataset.frame is frame, True)

    def test_adult_acquisition_normalizes_duplicate_test_file_labels(self) -> None:
        frame = _one_row_frame("adult")
        frame["income"] = ["  <=50K. "]
        with patch(
            "sbtab.benchmark.datasets.acquisition._fetch_uci_frame",
            return_value=frame,
        ):
            dataset = fetch_mixed_dataset("adult")

        self.assertEqual(dataset.frame["income"].tolist(), ["<=50K"])

    def test_fetch_unknown_key_does_not_enter_optional_source_code(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown"):
            fetch_mixed_dataset("unknown")


if __name__ == "__main__":
    unittest.main()

"""Explicit dataset declarations for the unified benchmark."""

from __future__ import annotations

from sbtab.benchmark.datasets.acquisition import (
    MIXED_DATASET_SOURCES,
    DatasetSource,
    fetch_all_mixed_datasets,
    fetch_mixed_dataset,
)
from sbtab.benchmark.datasets.bundle import (
    MIXED_DATASET_BUNDLE_VERSION,
    fetch_mixed_dataset_bundle,
    load_mixed_dataset_bundle,
    write_mixed_dataset_bundle,
)
from sbtab.benchmark.datasets.mixed import (
    MIXED_DATASET_COLUMNS,
    MIXED_DATASET_KEYS,
    make_mixed_dataset,
)
from sbtab.benchmark.datasets.online_shoppers import (
    ONLINE_SHOPPERS_COLUMNS,
    ONLINE_SHOPPERS_TARGET,
    ONLINE_SHOPPERS_UCI_ID,
    fetch_online_shoppers_frame,
    make_online_shoppers_dataset,
)

__all__ = [
    "MIXED_DATASET_COLUMNS",
    "MIXED_DATASET_BUNDLE_VERSION",
    "MIXED_DATASET_KEYS",
    "MIXED_DATASET_SOURCES",
    "ONLINE_SHOPPERS_COLUMNS",
    "ONLINE_SHOPPERS_TARGET",
    "ONLINE_SHOPPERS_UCI_ID",
    "DatasetSource",
    "fetch_all_mixed_datasets",
    "fetch_mixed_dataset",
    "fetch_mixed_dataset_bundle",
    "fetch_online_shoppers_frame",
    "make_mixed_dataset",
    "make_online_shoppers_dataset",
    "load_mixed_dataset_bundle",
    "write_mixed_dataset_bundle",
]

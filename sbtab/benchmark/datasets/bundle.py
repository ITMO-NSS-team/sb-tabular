"""Portable, checksum-verified offline bundles for mixed benchmark inputs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from uuid import uuid4

from sbtab.benchmark.artifacts import read_typed_table, write_typed_table
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.datasets.acquisition import (
    MIXED_DATASET_SOURCES,
    fetch_all_mixed_datasets,
)
from sbtab.benchmark.datasets.mixed import (
    MIXED_DATASET_KEYS,
    make_mixed_dataset,
)
from sbtab.benchmark.validation import ContractViolation


MIXED_DATASET_BUNDLE_VERSION = 1


def fetch_mixed_dataset_bundle(output_dir: Path) -> Path:
    """Acquire all fourteen datasets, then atomically write one offline bundle."""

    return write_mixed_dataset_bundle(fetch_all_mixed_datasets(), output_dir)


def write_mixed_dataset_bundle(
    datasets: tuple[TabularDataset, ...],
    output_dir: Path,
) -> Path:
    """Atomically create a portable collection from validated raw datasets."""

    if not isinstance(datasets, tuple) or not all(
        isinstance(dataset, TabularDataset) for dataset in datasets
    ):
        raise ContractViolation("datasets must be a tuple of TabularDataset values.")
    if not isinstance(output_dir, Path):
        raise ContractViolation("output_dir must be pathlib.Path.")
    keys = tuple(dataset.name for dataset in datasets)
    if len(set(keys)) != len(keys):
        raise ContractViolation("Dataset bundle contains duplicate dataset names.")
    unknown = tuple(key for key in keys if key not in MIXED_DATASET_KEYS)
    if unknown:
        raise ContractViolation(f"Dataset bundle contains unknown keys: {unknown!r}.")
    if output_dir.exists():
        raise ContractViolation(f"Dataset bundle directory already exists: {output_dir}.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(
        f".{output_dir.name}.initializing-{uuid4().hex}"
    )
    temporary.mkdir()
    try:
        entries: list[dict[str, object]] = []
        for dataset in datasets:
            # Rebuild through the declaration boundary so a caller cannot label
            # an unrelated TabularDataset with a canonical acquisition key.
            canonical = make_mixed_dataset(dataset.name, dataset.frame)
            table_path = temporary / f"{dataset.name}.json"
            digest = write_typed_table(canonical.frame, table_path)
            entries.append(
                {
                    "key": dataset.name,
                    "rows": len(canonical.frame),
                    "columns": list(canonical.column_order),
                    "table_path": table_path.name,
                    "table_sha256": digest,
                    "source": asdict(MIXED_DATASET_SOURCES[dataset.name]),
                }
            )
        manifest = {
            "artifact_type": "mixed_dataset_bundle",
            "artifact_version": MIXED_DATASET_BUNDLE_VERSION,
            "keys": list(keys),
            "datasets": entries,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except FileExistsError as error:
        raise ContractViolation(
            f"Dataset bundle directory was created concurrently: {output_dir}."
        ) from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / "manifest.json"


def load_mixed_dataset_bundle(
    bundle_dir: Path,
    *,
    keys: tuple[str, ...] | None = None,
) -> tuple[TabularDataset, ...]:
    """Load selected datasets after manifest, checksum, and schema validation."""

    if not isinstance(bundle_dir, Path):
        raise ContractViolation("bundle_dir must be pathlib.Path.")
    manifest_path = bundle_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractViolation(
            f"Dataset bundle manifest does not exist: {manifest_path}."
        ) from error
    except json.JSONDecodeError as error:
        raise ContractViolation("Dataset bundle manifest is not valid JSON.") from error
    if not isinstance(manifest, dict) or (
        manifest.get("artifact_type") != "mixed_dataset_bundle"
        or manifest.get("artifact_version") != MIXED_DATASET_BUNDLE_VERSION
    ):
        raise ContractViolation("Dataset bundle has an unsupported type or version.")
    entries = manifest.get("datasets")
    manifest_keys = manifest.get("keys")
    if not isinstance(entries, list) or not isinstance(manifest_keys, list):
        raise ContractViolation("Dataset bundle manifest structure is invalid.")
    by_key = {
        entry.get("key"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("key"), str)
    }
    if tuple(by_key) != tuple(manifest_keys) or len(by_key) != len(entries):
        raise ContractViolation("Dataset bundle keys are duplicated or inconsistent.")

    selected = tuple(manifest_keys) if keys is None else keys
    if not isinstance(selected, tuple) or not all(
        isinstance(key, str) for key in selected
    ):
        raise ContractViolation("keys must be a tuple of dataset names or None.")
    if len(set(selected)) != len(selected):
        raise ContractViolation("Requested dataset keys contain duplicates.")
    missing = tuple(key for key in selected if key not in by_key)
    if missing:
        raise ContractViolation(f"Dataset bundle lacks requested keys: {missing!r}.")

    result: list[TabularDataset] = []
    for key in selected:
        entry = by_key[key]
        table_name = entry.get("table_path")
        digest = entry.get("table_sha256")
        if not isinstance(table_name, str) or not isinstance(digest, str):
            raise ContractViolation(f"Dataset bundle entry {key!r} is invalid.")
        frame = read_typed_table(bundle_dir / table_name, expected_sha256=digest)
        dataset = make_mixed_dataset(key, frame)
        if entry.get("rows") != len(frame) or entry.get("columns") != list(
            dataset.column_order
        ):
            raise ContractViolation(
                f"Dataset bundle entry {key!r} row or column metadata differs."
            )
        result.append(dataset)
    return tuple(result)

"""Create-only final-evaluation artifacts linked to generation evidence."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from sbtab.benchmark.artifacts import cross_validation_result_fingerprint
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation.final import CrossValidationEvaluation


EVALUATION_ARTIFACT_VERSION = 2


def _read_generation_manifest(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise ContractViolation(
            f"Generation manifest does not exist: {path}."
        )
    payload = path.read_bytes()
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ContractViolation(
            f"Generation manifest is not valid JSON: {path}."
        ) from error
    if not isinstance(manifest, dict):
        raise ContractViolation("Generation manifest root must be an object.")
    digest = hashlib.sha256(payload).hexdigest()
    return manifest, digest


def _validate_generation_link(
    result: CrossValidationEvaluation,
    manifest: dict[str, object],
) -> None:
    generation = result.generation
    if manifest.get("artifact_type") != "cross_validation_generation" or (
        manifest.get("artifact_version") != 2
    ):
        raise ContractViolation(
            "Evaluation must reference a cross-validation generation manifest."
        )
    if manifest.get("adapter_name") != generation.adapter_name:
        raise ContractViolation(
            "Generation manifest adapter does not match evaluation result."
        )
    if manifest.get("dataset") != generation.dataset.name:
        raise ContractViolation(
            "Generation manifest dataset does not match evaluation result."
        )
    if manifest.get("result_fingerprint") != (
        cross_validation_result_fingerprint(generation)
    ):
        raise ContractViolation(
            "Generation manifest content does not match evaluation result."
        )
    manifest_folds = manifest.get("folds")
    if not isinstance(manifest_folds, list):
        raise ContractViolation("Generation manifest folds must be a list.")
    manifest_fold_ids = tuple(
        fold.get("fold_id")
        for fold in manifest_folds
        if isinstance(fold, dict)
    )
    result_fold_ids = tuple(fold.fold_id for fold in result.folds)
    if manifest_fold_ids != result_fold_ids:
        raise ContractViolation(
            "Generation manifest folds do not match evaluation result."
        )


def write_evaluation_artifacts(
    result: CrossValidationEvaluation,
    output_dir: Path,
    *,
    generation_manifest: Path,
) -> Path:
    """Write final fold metrics and a manifest linked to generation bytes.

    Parameters
    ----------
    result:
        Completed final evaluation of one in-memory generation result.
    output_dir:
        New create-only destination for `metrics.json` and `manifest.json`.
    generation_manifest:
        Existing generation manifest for the same result. Its relative path
        and SHA-256 digest are stored so reviewers can resolve exact inputs.

    Returns
    -------
    Path
        Evaluation manifest written last after metrics succeed.
    """

    if not isinstance(result, CrossValidationEvaluation):
        raise ContractViolation("result must be CrossValidationEvaluation.")
    if not isinstance(output_dir, Path):
        raise ContractViolation("output_dir must be pathlib.Path.")
    if not isinstance(generation_manifest, Path):
        raise ContractViolation("generation_manifest must be pathlib.Path.")

    source_manifest, source_digest = _read_generation_manifest(
        generation_manifest
    )
    _validate_generation_link(result, source_manifest)
    metrics = {
        "summary": asdict(result.summary),
        "folds": [asdict(fold) for fold in result.folds],
    }
    metrics_bytes = (
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    manifest = {
        "artifact_type": "cross_validation_evaluation",
        "artifact_version": EVALUATION_ARTIFACT_VERSION,
        "adapter_name": result.generation.adapter_name,
        "dataset": result.generation.dataset.name,
        "folds": len(result.folds),
        "generation_manifest": os.path.relpath(
            generation_manifest,
            start=output_dir,
        ),
        "generation_manifest_sha256": source_digest,
        "metrics_path": "metrics.json",
        "utility_metric": (
            result.summary.utility.metric.value
            if result.summary.utility is not None
            else None
        ),
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    if output_dir.exists():
        raise ContractViolation(
            f"Evaluation artifact directory already exists: {output_dir}."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(
        f".{output_dir.name}.initializing-{uuid4().hex}"
    )
    temporary.mkdir()
    try:
        (temporary / "metrics.json").write_bytes(metrics_bytes)
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        temporary.replace(output_dir)
    except FileExistsError as error:
        raise ContractViolation(
            f"Evaluation artifact directory already exists: {output_dir}."
        ) from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / "manifest.json"

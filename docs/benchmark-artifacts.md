# Benchmark artifacts and restart protocol

Status: normative v1 persistence boundary for final cross-validation.

## Purpose

Final model runs can take hours. A process interruption after fold 3 must not
discard three completed fits, and a restart must never silently combine folds
from different datasets, splits, seeds, or model families.

`run_cross_validation_resumable(..., output_dir=...)` implements this boundary.
It prepares the same model-independent `CrossValidationPlan` as the in-memory
runner, then opens or creates a `GenerationArtifactStore` bound to that plan.

## Directory lifecycle

The store creates this layout:

```text
generation/
├── plan.json
├── real-post-policy.json
├── fold-0/
│   ├── fold.json
│   └── synthetic.json
├── fold-1/
│   ├── fold.json
│   └── synthetic.json
└── manifest.json
```

`plan.json` fingerprints dataset declarations and exact post-policy values,
missing-policy evidence, positional splits, seeds, device, and all other
shared runtime controls. Reusing the directory with any changed input fails
before an adapter is constructed.

Each `fold-N` directory is committed atomically after fit, sample, validation,
and decode succeed. `fold.json` is its commit marker and records the plan
fingerprint, adapter name, exact split, row counts, timings, and SHA-256 of the
typed synthetic table. A directory missing its commit marker or containing a
checksum mismatch is corruption, not an incomplete fold to retrain silently.

On restart, committed folds are decoded and validated against the current
plan. Only absent folds construct fresh adapters and train. `manifest.json` is
written after every planned fold is verified; writing it is idempotent for the
same result and refuses conflicting content.

Temporary sibling directories have `.initializing-` or `.tmp-` in their names
and are never treated as evidence. They may remain after an operating-system
level termination and can be inspected or removed by the operator after the
run is no longer active.

## Typed table format

Truth data uses strict tagged JSON scalars instead of pandas pickle. This
avoids the cross-version pickle failure that occurred when moving datasets
between macOS, Kaggle, and the GPU virtual machine. The format distinguishes
null, boolean, integer, finite float, string, pandas timestamp, and pandas
timedelta values. Unsupported exotic categorical scalars fail explicitly at
artifact creation.

CSV may be exported separately for human convenience, but it is not restart
evidence because CSV cannot preserve categorical scalar identity reliably.
Generated artifacts are local run outputs and must not be committed to source
branches.

## Evaluation linkage

`write_evaluation_artifacts` creates its output directory atomically. Its
manifest stores the relative generation-manifest path and SHA-256 digest. It
also verifies a deterministic result fingerprint covering the exact plan,
adapter name, synthetic fold tables, and native fit/sample timings. A manifest
with matching dataset and fold labels but different bytes is rejected.

Evaluation output is create-only. Re-running evaluation must use a new output
directory, preserving prior evidence for review.

## Verification

From the repository root:

```bash
python -m unittest \
  tests.benchmark.test_runner \
  tests.benchmark.test_artifacts \
  tests.benchmark.test_final_evaluation
```

The tests simulate interruption after one fold, resume without retraining that
fold, repeat an already completed run, reject a changed plan and changed raw
values, detect checksum corruption, and verify the generation/evaluation link.

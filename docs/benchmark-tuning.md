# Reusable Optuna tuning layer

The benchmark exposes one reusable Phase-A Optuna lifecycle in
`sbtab.benchmark.tuning`. A model tuner supplies a typed native configuration
and a few small callbacks; the shared layer runs the canonical holdout,
calculates the raw-space tuning objective, tracks successful and failed
attempts, protects persistent studies from incompatible resume, and restores
the selected native configuration.

The fixed-configuration runner and native models remain independent of Optuna.
The tuning module is called only by model-owned tuning entrypoints.

## Ownership boundary

The shared layer owns:

- creation and compatible resume of an Optuna study;
- a deterministic study fingerprint covering data, semantics, holdout, seeds,
  device, model identity, sampler seed, and versioned protocols;
- the successful-trial target and total-attempt safety ceiling;
- one canonical `run_holdout_trial` and `evaluate_tuning_score` call per trial;
- common score, timing, missing-policy, and failure evidence;
- reversible reconstruction of the best typed native configuration;
- create-only portable Phase-A review artifacts.

Each model integration owns:

- the typed native configuration and Optuna search space;
- construction of its thin adapter from that configuration;
- JSON encoding and decoding of the native configuration;
- classification of known model-specific numerical failures;
- any explicitly justified later selection stage, such as expensive multi-seed
  reranking of several TabDDPM candidates.

Search spaces and numerical exceptions are model mathematics and must not be
encoded as model-name branches in the shared layer.

## Minimal model integration

```python
from dataclasses import asdict

import optuna

from sbtab.benchmark.runner import HoldoutRunConfig
from sbtab.benchmark.tuning import (
    OptunaStudyConfig,
    run_optuna_holdout_study,
)


def suggest_config(trial: optuna.Trial) -> MyModelConfig:
    return MyModelConfig(
        learning_rate=trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-3,
            log=True,
        ),
        batch_size=trial.suggest_categorical(
            "batch_size",
            [256, 512, 1024],
        ),
    )


def decode_config(payload: object) -> MyModelConfig:
    if not isinstance(payload, dict):
        raise TypeError("MyModelConfig payload must be an object")
    return MyModelConfig(**payload)


result = run_optuna_holdout_study(
    dataset,
    model_name=MyModelAdapter.name,
    input_spec=MyModelAdapter.input_spec,
    config=OptunaStudyConfig(
        run=HoldoutRunConfig(...),
        target_complete_trials=30,
        max_total_trials=45,
        sampler_seed=5,
        study_name="my-model-adult-phase-a-v1",
        storage="sqlite:///study.sqlite3",
        load_if_exists=True,
        protocol_version=1,
        objective_version=1,
    ),
    suggest_config=suggest_config,
    make_adapter=MyModelAdapter,
    encode_config=asdict,
    decode_config=decode_config,
    classify_failure=classify_known_numerical_failure,
)
```

`target_complete_trials` is the desired total number of successful trials in
the study, not the number to add on resume. `max_total_trials` counts every
stored trial, including failed and pruned attempts. Re-running with the same
SQLite storage therefore adds only missing successful trials.

## Failure policy

A classifier may return `TrialFailure(PRUNE, ...)` for a known unfavorable but
valid numerical region, `TrialFailure(FAIL, ...)` for an expected failed
attempt, or `TrialFailure(RAISE, ...)` to stop after recording the failure. An
explicit `optuna.TrialPruned` is recorded as `PRUNE` automatically. Any other
unclassified exception is recorded as failed and re-raised. The shared layer
never hides an unknown implementation or contract error.

The classifier accepts `BaseException` because some existing native numerical
errors use that legacy base class. Process-control signals such as
`KeyboardInterrupt`, `SystemExit`, and task cancellation are never classified;
they stop immediately and leave persistent-study recovery to the next run.

If a process exits while a persistent trial is `RUNNING`, the next compatible
resume marks that abandoned trial `FAIL` before continuing. Interrupting a run
does not produce a misleading partial success artifact.

## Review artifacts

`write_optuna_tuning_artifacts(result, output_dir)` atomically creates:

- `best-config.json` with the selected native configuration;
- `trials.json` with parameters, states, values, and standardized evidence;
- `manifest.json` with dataset, semantic views, study counts, holdout controls,
  fingerprint, and links to the other files.

The destination is create-only. The Optuna storage URI is intentionally not
serialized because it may contain credentials. A later model-specific rerank
must write a separate linked artifact rather than modifying Phase A in place.

## Verification

From the repository root:

```bash
python -m unittest tests.benchmark.test_optuna_tuning
```

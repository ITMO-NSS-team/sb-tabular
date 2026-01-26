# sb-tabular

This repository contains an experimental framework for **Schrödinger Bridges (SB)** applied to
**tabular data**. 

---

## Project tree

```text
sb-tabular/
├── README.md                                  # Main project documentation (overview, quickstart, structure)
├── LICENSE                                    # License (add if not present; important for OSS/public use)
├── pyproject.toml                             # (recommended) Packaging + dependencies (or setup.cfg/requirements.txt)
├── requirements.txt                           # (optional) Simple dependency list for dev / CI
├── .gitignore                                 # Git ignore rules
├── .pre-commit-config.yaml                    # (recommended) Lint/format hooks (ruff/black/isort/etc.)
├── .github/                                   # (recommended) GitHub workflows and templates
│   └── workflows/
│       ├── tests.yml                          # (recommended) CI: run unit tests, linting
│       └── build.yml                          # (optional) CI: package build checks
│
├── sbtab/                                     # Core library package (import as `sbtab`)
│   ├── __init__.py                            # Public API exports (keep minimal, stable)
│   │
│   ├── core/                                  # (recommended) Shared utilities for reproducibility
│   │   
│   ├── data/                                  # Dataset schema + splitting + datamodule
│   │   ├── __init__.py
│   │   ├── schema.py                          # TabularSchema (continuous-only): column lists + validation
│   │   ├── splits.py                          # Splitting protocols:
│   │   │                                        # - K-fold splits for experiments
│   │   │                                        # - Single holdout split for tuning generative models
│   │   └── datamodule.py                      # TabularDataModule:
│   │                                            # - global missing removal
│   │                                            # - provides k-fold/holdout slices
│   │                                            # - refits transforms on train portion per split
│   │
│   ├── transforms/                            # Preprocessing pipeline (continuous-only at this stage)
│   │   ├── __init__.py
│   │   ├── base.py                            # BaseTransform protocol + TransformState serialization
│   │   ├── missing.py                         # DropMissingRows (fixed missing strategy: drop rows with NaNs)
│   │   ├── continuous.py                      # ContinuousStandardScaler (fixed scaling: standardize)
│   │   └── pipeline.py                        # TransformPipeline:
│   │                                            # - sequential transforms
│   │                                            # - pre-fit transform allowed for missing removal
│   │                                            # - cloneable for fold-wise refitting
│   │
│   ├── bridge/                                # Schrödinger bridge primitives (solver-agnostic)
│   │   ├── __init__.py
│   │   ├── timegrid.py                        # TimeGrid: discretized time axis t_k, steps γ_k, total T
│   │   ├── reference.py                       # Reference endpoint sampling (currently Gaussian N(0, I))
│   │   ├── sde.py                             # Euler–Maruyama integrator (stochastic/deterministic stepping)
│   │   ├── pathsampler.py                     # Simulate forward/backward trajectories on TimeGrid
│   │   └── losses.py                          # Generic losses (RegressionLoss: MSE/Huber) used in cache training
│   │
│   ├── models/                                # Learnable components (fields/drifts/embeddings)
│   │   ├── __init__.py
│   │   └── field/                             # Field models: (x, t) -> vector field in data space
│   │       ├── __init__.py
│   │       ├── neural/                        # Neural field models
│   │       │   ├── __init__.py
│   │       │   ├── time_embedding.py          # Sinusoidal time embedding for scalar t
│   │       │   ├── mlp.py                     # TimeConditionedMLP: MLP(x, t) -> R^D drift/field
│   │       │   └── trainer.py                 # NeuralTrainer: minimal training loop wrapper
│   │       │
│   │       └── boosted/                       # (planned) Boosting models per time step
│   │           ├── __init__.py
│   │           ├── per_step.py                # (planned) Per-step booster list: model_k for each step k
│   │           ├── lgbm.py                     # (planned) LightGBM wrapper (fit/predict per step)
│   │           ├── xgb.py                      # (planned) XGBoost wrapper (fit/predict per step)
│   │           └── catboost.py                 # (planned) CatBoost wrapper (fit/predict per step)
│   │
│   ├── generative/                            # (recommended) Unified generative model interface + adapters
│   │   ├── __init__.py
│   │   ├── base.py                            # (recommended) GenerativeModel API: fit/sample/save/load
│   │   └── wrappers.py                        # (recommended) wrappers for external baseline libraries
│   │
│   ├── solvers/                               # SB solvers (each in its own submodule)
│   │   ├── __init__.py
│   │   ├── base.py                            # (recommended) BaseSBSolver: shared hooks, logging, save/load
│   │   │
│   │   ├── ipf_dsb/                           # IPF + DSB solver (integrated from diffusion_schrodinger_bridge)
│   │   │   ├── __init__.py
│   │   │   └── solver.py                      # IPF loop:
│   │   │                                        # - forward net f, backward net b
│   │   │                                        # - cache-based regression training
│   │   │                                        # - generation by backward dynamics from Gaussian prior
│   │   │
│   │   ├── imf_dsbm/                           # (planned) IMF + DSBM solver implementation
│   │   │   ├── __init__.py
│   │   │   └── solver.py                      # (planned) IMF training loop + DSBM objective
│   │   │
│   │   ├── light_sb/                           # (planned) Light Schrödinger Bridge solver
│   │   │   ├── __init__.py
│   │   │   └── solver.py                      # (planned) LightSB training procedure
│   │   │
│   │   └── asbm/                               # (planned) ASBM / adaptive solvers
│   │       ├── __init__.py
│   │       └── solver.py                      # (planned) ASBM approach (adaptive schedules, regularizers)
│   │
│   ├── baselines/                              # (recommended) Non-SB generative baselines under same API
│   │   ├── __init__.py
│   │   ├── ctgan/                              # (recommended) CTGAN wrapper
│   │   │   ├── __init__.py
│   │   │   └── model.py                       # Fit/sample adapter
│   │   ├── tabddpm/                            # (recommended) TabDDPM wrapper
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   ├── tabsyn/                             # (recommended) TabSyn wrapper
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   └── stasy/                              # (recommended) STaSy wrapper
│   │       ├── __init__.py
│   │       └── model.py
│   │
│   ├── evaluation/                             # (recommended) Evaluation metrics + reporting
│   │   ├── __init__.py
│   │   ├── evaluator.py                        # Orchestrates metric computation on real vs synthetic
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── statistical.py                  # WD/MMD/marginal KL/corr-Frobenius, etc.
│   │   │   ├── utility.py                      # TSTR/TRTS, CatBoost R2/MAPE, etc.
│   │   │   └── privacy.py                      # (optional) privacy checks
│   │   └── reports/
│   │       ├── __init__.py
│   │       ├── tables.py                       # Export CSV/LaTeX summary tables
│   │       └── plots.py                        # Plotting helpers
│   │
│   └── experiments/                            # (recommended) Experiment runner + sweeps
│       ├── __init__.py
│       ├── runner.py                           # Train->sample->evaluate pipeline for one config
│       ├── sweeps.py                           # Grid/random/optuna sweep utilities
│       └── tracking.py                         # Run directory structure + metadata saving
│
├── examples/                                   # End-to-end demos (runnable scripts)
│   ├── California_Housing_example.py           # Train IPF+DSB on CA housing (holdout split), sample + save CSV
│   └── ...                                     # Additional examples (k-fold, ablations, etc.)
│
├── scripts/                                    # (recommended) Shell scripts for reproducible runs
│   ├── run_california_holdout.sh               # Example entrypoint for training on CA housing
│   ├── run_kfold_benchmark.sh                  # Benchmark mode: k-fold evaluation for solvers/baselines
│   └── make_tables.sh                          # Generate tables/figures from saved metrics
│
├── configs/                                    # (recommended) Hydra/OmegaConf YAMLs for experiments
│   ├── dataset/                                # Dataset configs (paths, columns, targets)
│   ├── transform/                              # Transform pipeline configs
│   ├── solver/                                 # SB solver configs (ipf_dsb, light_sb, etc.)
│   ├── model_field/                            # Field model configs (neural mlp, boosted per-step)
│   ├── baseline/                               # Baseline configs (ctgan, tabddpm, etc.)
│   ├── eval/                                   # Metric bundles
│   └── experiment/                             # Experiment presets (compare_all, ablations)
│
├── tests/                                      # (recommended) Unit and smoke tests
│   ├── test_schema.py                          # Schema validation tests
│   ├── test_splits.py                          # K-fold and holdout split tests (independent protocols)
│   ├── test_transforms_roundtrip.py            # Scaler inverse_transform consistency
│   ├── test_ipf_dsb_smoke.py                   # Short fit/sample run for IPF+DSB
│   └── test_pathsampler.py                     # Path sampling direction and shape tests
│
└── notebooks/                                  # (optional) Interactive analyses / debugging
    ├── demo_sampling.ipynb                     # Visual sanity checks, distributions, correlations
    └── results_analysis.ipynb                  # Aggregate metrics across runs

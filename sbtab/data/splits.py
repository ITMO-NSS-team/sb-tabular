
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import KFold, train_test_split


@dataclass(frozen=True)
class SplitConfigKFold:
    """Config for k-fold splitting used in experiments."""
    n_splits: int = 5
    shuffle: bool = True
    random_seed: int = 42


@dataclass(frozen=True)
class SplitConfigHoldout:
    """Config for a single train/val split used for tuning generative models."""
    val_size: float = 0.2
    shuffle: bool = True
    random_seed: int = 42


@dataclass(frozen=True)
class KFoldSplit:
    """One CV fold split (positional indices)."""
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class HoldoutSplit:
    """One fixed train/val split (positional indices)."""
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_kfold_splits(n_samples: int, cfg: SplitConfigKFold) -> List[KFoldSplit]:
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1")
    if cfg.n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    all_idx = np.arange(n_samples)
    kf = KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_seed)

    folds: List[KFoldSplit] = []
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(all_idx)):
        folds.append(
            KFoldSplit(
                fold_id=fold_id,
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
            )
        )
    return folds


def make_holdout_split(n_samples: int, cfg: SplitConfigHoldout) -> HoldoutSplit:
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1")
    if not (0.0 < cfg.val_size < 1.0):
        raise ValueError("val_size must be in (0, 1)")

    all_idx = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=cfg.val_size,
        shuffle=cfg.shuffle,
        random_state=cfg.random_seed,
    )
    return HoldoutSplit(
        train_idx=np.asarray(train_idx, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
    )

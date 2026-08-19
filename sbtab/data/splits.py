from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


@dataclass(frozen=True)
class SplitConfigKFold:
    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class KFoldSplit:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


class SafeCategoricalKFold:
    """
    KFold с гарантией, что в train каждого фолда попадают ВСЕ уникальные
    категории из указанных cat_columns.

    - Если y передан — базовый сплит делается через StratifiedKFold.
    - Если y is None — через обычный KFold.
    - После базового сплита «редкие» категории, оказавшиеся только в test,
      переносятся в train (по одной строке на каждую пропущенную категорию).
    """

    def __init__(self, cat_columns, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.cat_columns = cat_columns
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is not None:
            base_kf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
            )
            splitter = base_kf.split(X, y)
        else:
            base_kf = KFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
            )
            splitter = base_kf.split(X)

        for train_idx, test_idx in splitter:
            train_idx = list(train_idx)
            test_idx = list(test_idx)

            for col in self.cat_columns:
                if col not in X.columns:
                    continue
                unique_in_full = set(X[col].dropna().unique())
                unique_in_train = set(X.iloc[train_idx][col].unique())
                missing_cats = unique_in_full - unique_in_train

                for cat in missing_cats:
                    missing_indices = [
                        idx for idx in test_idx if pd.notna(X.iloc[idx][col]) and X.iloc[idx][col] == cat
                    ]
                    if missing_indices:
                        idx_to_move = missing_indices[0]
                        train_idx.append(idx_to_move)
                        test_idx.remove(idx_to_move)

            yield np.array(train_idx), np.array(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

@dataclass(frozen=True)
class SplitConfigHoldout:
    val_size: float = 0.2
    shuffle: bool = True
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class HoldoutSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_kfold_splits(
    n_samples: int,
    cfg: SplitConfigKFold,
    labels: Optional[np.ndarray] = None
) -> List[KFoldSplit]:
    if cfg.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_samples < cfg.n_splits:
        raise ValueError("n_samples must be greater than or equal to n_splits.")

    if labels is not None:
        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state)
        splits: List[KFoldSplit] = []
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_samples), labels)):
            splits.append(KFoldSplit(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx))
        return splits

    indices = np.arange(n_samples)
    if cfg.shuffle:
        rng = np.random.default_rng(cfg.random_state)
        indices = rng.permutation(indices)

    fold_sizes = np.full(cfg.n_splits, n_samples // cfg.n_splits, dtype=int)
    fold_sizes[: n_samples % cfg.n_splits] += 1

    splits: List[KFoldSplit] = []
    current = 0
    for fold_id, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append(KFoldSplit(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx))
        current = stop
    return splits


def make_holdout_split(n_samples: int, cfg: SplitConfigHoldout) -> HoldoutSplit:
    if not 0.0 < cfg.val_size < 1.0:
        raise ValueError("val_size must be in the open interval (0, 1).")
    if n_samples < 2:
        raise ValueError("At least 2 samples are required to create a holdout split.")

    indices = np.arange(n_samples)
    if cfg.shuffle:
        rng = np.random.default_rng(cfg.random_state)
        indices = rng.permutation(indices)

    n_val = int(round(n_samples * cfg.val_size))
    n_val = min(max(n_val, 1), n_samples - 1)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return HoldoutSplit(train_idx=train_idx, val_idx=val_idx)

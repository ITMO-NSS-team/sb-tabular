
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd

from .schema import TabularSchema
from .splits import (
    SplitConfigKFold,
    SplitConfigHoldout,
    KFoldSplit,
    HoldoutSplit,
    make_kfold_splits,
    make_holdout_split,
)


@dataclass
class FoldData:
    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class HoldoutData:
    train: pd.DataFrame
    val: pd.DataFrame


class TabularDataModule:
    """
    Continuous-only data module.

    Provides TWO INDEPENDENT splitting protocols:
      - k-fold splits for experiments
      - single holdout split for tuning generative models

    Transforms (e.g., DropMissingRows + StandardScaler) are applied consistently:
      - missing rows removed BEFORE any splitting
      - scaler fitted on the corresponding *train* subset for each requested split
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: TabularSchema,
        transforms: Optional[Any] = None,
        reset_index: bool = True,
    ) -> None:
        self.schema = schema
        self.transforms = transforms

        schema.validate(df)
        df0 = df.copy()

        # Global missing-value removal must happen before any split.
        # If transforms include DropMissingRows, calling transform is enough;
        # otherwise user should pass already-clean df.
        if self.transforms is not None:
            try:
                df0 = self.transforms.transform(df0)
            except Exception:
                # If pipeline requires fit, do minimal safe behavior:
                # fit+transform on full df; fold-wise re-fit will be done later anyway.
                self.transforms.fit(df0, self.schema)
                df0 = self.transforms.transform(df0)

        if reset_index:
            df0 = df0.reset_index(drop=True)

        self.df_clean = df0
        self.n_samples = len(df0)

        # Cached splits (optional; computed lazily if you prefer)
        self._kfold_splits: Optional[list[KFoldSplit]] = None
        self._holdout_split: Optional[HoldoutSplit] = None

    # --------- K-FOLD (experiments) ---------

    def prepare_kfold(self, cfg: SplitConfigKFold) -> None:
        self._kfold_splits = make_kfold_splits(self.n_samples, cfg)

    def get_fold(self, fold_id: int) -> FoldData:
        if self._kfold_splits is None:
            raise RuntimeError("K-fold splits are not prepared. Call prepare_kfold(cfg) first.")

        if fold_id < 0 or fold_id >= len(self._kfold_splits):
            raise IndexError(f"fold_id={fold_id} out of range (n_folds={len(self._kfold_splits)})")

        fold = self._kfold_splits[fold_id]
        train_raw = self.df_clean.iloc[fold.train_idx].copy()
        test_raw = self.df_clean.iloc[fold.test_idx].copy()

        if self.transforms is None:
            return FoldData(fold_id=fold_id, train=train_raw, test=test_raw)

        pipe = self._clone_transforms(self.transforms)
        pipe.fit(train_raw, self.schema)  # fit scaler only on fold-train
        return FoldData(
            fold_id=fold_id,
            train=pipe.transform(train_raw),
            test=pipe.transform(test_raw),
        )

    def get_all_folds(self) -> Dict[int, FoldData]:
        if self._kfold_splits is None:
            raise RuntimeError("K-fold splits are not prepared. Call prepare_kfold(cfg) first.")
        return {f.fold_id: self.get_fold(f.fold_id) for f in self._kfold_splits}

    # --------- HOLDOUT (tuning) ---------

    def prepare_holdout(self, cfg: SplitConfigHoldout) -> None:
        self._holdout_split = make_holdout_split(self.n_samples, cfg)

    def get_holdout(self) -> HoldoutData:
        if self._holdout_split is None:
            raise RuntimeError("Holdout split is not prepared. Call prepare_holdout(cfg) first.")

        sp = self._holdout_split
        train_raw = self.df_clean.iloc[sp.train_idx].copy()
        val_raw = self.df_clean.iloc[sp.val_idx].copy()

        if self.transforms is None:
            return HoldoutData(train=train_raw, val=val_raw)

        pipe = self._clone_transforms(self.transforms)
        pipe.fit(train_raw, self.schema)  # fit scaler only on holdout-train
        return HoldoutData(
            train=pipe.transform(train_raw),
            val=pipe.transform(val_raw),
        )

    # --------- utils ---------

    @staticmethod
    def _clone_transforms(transforms: Any) -> Any:
        """
        Best-effort cloning:
          - If pipeline supports get_state/from_state -> use it.
          - Else deepcopy.
        """
        if hasattr(transforms, "get_state") and hasattr(transforms.__class__, "from_state"):
            state = transforms.get_state()
            return transforms.__class__.from_state(state)  # type: ignore[attr-defined]
        import copy
        return copy.deepcopy(transforms)

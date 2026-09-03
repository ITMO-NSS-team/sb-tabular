from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema, classify_feature_type
from sbtab.evaluation.cross_validator import CrossValidator
from sbtab.transforms.drop_cols import DropDataCols
from sbtab.transforms.missing import DropMissingRows
from sbtab.transforms.pipeline import TransformPipeline

CONFIG = {
    "model": "msbm",
    "pickle": "../data/datasets/datasets_mixed.pkl",
    "results_dir": "../experiments/tuning_script/msbm_optuna_results",
    "output": "cv_msbm_results.csv",
    "datasets": "all",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def main():
    model = CONFIG["model"]
    pickle_path = CONFIG["pickle"]
    results_dir = Path(CONFIG["results_dir"])
    output_csv = CONFIG["output"]
    datasets_str = CONFIG["datasets"]
    device = CONFIG["device"]

    with open(pickle_path, "rb") as f:
        all_data = pickle.load(f)

    dataset_keys = (
        list(all_data.keys())
        if datasets_str.lower() == "all"
        else [k.strip() for k in datasets_str.split(",")]
    )
    all_cv_results = []

    for ds_name in dataset_keys:
        print(f"\nEvaluating {model} on {ds_name}")
        df_raw = all_data[ds_name]

        target_col = df_raw.attrs.get("target_variable")
        task_type = df_raw.attrs.get("task_type", "classification")

        param_file = results_dir / f"{ds_name}_final_metrics.json"
        train_pkl = results_dir / f"{ds_name}_train.pkl"

        if not param_file.exists():
            print(f"  Skipping {ds_name}: {param_file.name} not found")
            continue
        if not train_pkl.exists():
            print(f"  Skipping {ds_name}: {train_pkl.name} not found")
            continue

        with open(param_file) as f:
            tuning_info = json.load(f)
        best_params = tuning_info["best_params"]

        with open(train_pkl, "rb") as f:
            train_df = pickle.load(f)

        schema = TabularSchema.infer_from_dataframe(train_df, target_col=target_col)

        dm = TabularDataModule(
            df=train_df,
            schema=schema,
            transforms=TransformPipeline(transforms=[DropMissingRows(), DropDataCols()]),
        )

        pure_cat_cols = list(schema.categorical_cols)
        discrete_cols = list(schema.discrete_cols)
        num_cols = list(schema.continuous_cols)

        if target_col:
            target_col_type = classify_feature_type(df_raw[target_col])
            if target_col_type == "continuous":
                num_cols.append(target_col)
            elif target_col_type == "discrete":
                discrete_cols.append(target_col)
            elif target_col_type == "categorical":
                pure_cat_cols.append(target_col)
            else:
                raise ValueError("Target column type not recognized")

        cv = CrossValidator(
            model_type=model,
            best_params=best_params,
            ds_name=ds_name,
            target_col=target_col,
            task_type=task_type,
            device=device,
            seed=42,
            k_folds=5,
        )

        df_fold_results = cv.run(dm, pure_cat_cols, discrete_cols, num_cols)
        all_cv_results.append(df_fold_results)

    if all_cv_results:
        final_df = pd.concat(all_cv_results, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        CrossValidator.summarise_cv(output_csv, "mean_cv_msbm_results.csv")
        print(f"\nSaved results to {output_csv}")


if __name__ == "__main__":
    main()
# Mixed benchmark datasets

Status: semantically reviewed v1 declaration for the fourteen-dataset set.

## Scope

The upstream repository publishes fourteen mixed datasets in
`sbtab/data/datasets/datasets_mixed.pkl` and constructs them in
`sbtab/data/get_datasets.py` in
[dataset update commit `33850fc`](https://github.com/Anaxagor/sb-tabular/commit/33850fc4805508fca993084634ef40e308dfa627).
The pickle stores raw frames plus feature groups inferred by the legacy
`TabularSchema` in `DataFrame.attrs`. Those inferred groups are acquisition
evidence, not the new contract: numeric dtype and cardinality cannot determine
whether a value is a magnitude, an ordered count, or a nominal code.

New benchmark code does not consume those attributes or infer semantics.
`sbtab/benchmark/datasets/mixed.py` declares each `ColumnSpec` from the meaning
of the field. Every classification target is an ordinary categorical modeled
column; every regression target is an ordinary continuous modeled column.

Online Shoppers is the intentional exception to a literal metadata migration:
it reuses the already approved pilot declaration. In particular, `Revenue` is
the categorical target, the three page-count columns are discrete, and numeric
nominal codes such as `TrafficType` are categorical.

## Collection

| Key | Published name | Source | Exact table | Target | Task | Removed before declaration |
| --- | --- | --- | --- | --- | --- | --- |
| `adult` | Adult | UCI 2 | features + target | `income` | classification | — |
| `credit_approval` | Credit Approval | UCI 27 | features + target | `A16` | classification | — |
| `online_shoppers` | Online Shoppers | UCI 468 | features + target | `Revenue` | classification | — |
| `eucalyptus` | Eucalyptus | OpenML 188 | OpenML frame | `Utility` | classification | — |
| `forest_fires` | Forest Fires | UCI 162 | features + target | `area` | regression | — |
| `insurance` | Insurance | Kaggle `mirichoi0218/insurance` | `insurance.csv` | `charges` | regression | — |
| `house_sales` | House Sales | Kaggle `harlfoxem/housesalesprediction` | `kc_house_data.csv` | `price` | regression | `id` |
| `cardiovascular_disease` | Cardiovascular Disease | Kaggle `sulianova/cardiovascular-disease-dataset` | `cardio_train.csv` | `cardio` | classification | `id` |
| `churn_modelling` | Churn Modelling | Kaggle `shrutimechlearn/churn-modelling` | `Churn_Modelling.csv` | `Exited` | classification | `RowNumber`, `CustomerId`, `Surname` |
| `auto_mpg` | Auto MPG | UCI 9 | features + target | `mpg` | regression | — |
| `diamonds` | Diamonds | Kaggle `shivam2503/diamonds` | `diamonds.csv` | `price` | regression | `Unnamed: 0` |
| `real_estate` | Real Estate | Kaggle `quantbruce/real-estate-price-prediction` | `Real estate.csv` | `Y house price of unit area` | regression | `No` |
| `stroke_prediction` | Stroke Prediction | Kaggle `fedesoriano/stroke-prediction-dataset` | `healthcare-dataset-stroke-data.csv` | `stroke` | classification | `id` |
| `palmer_penguins` | Palmer Penguins | Kaggle `parulpandey/palmer-archipelago-antarctica-penguin-data` | `penguins_lter.csv` | `Species` | classification | `studyName`, `Sample Number`, `Individual ID`, `Region`, `Stage`, `Comments` |

The earlier metrics table contains twelve of these datasets. `House Sales` and
`Diamonds` are the additional two entries in the current published collection.
Bike Sharing appeared briefly in an intermediate update and is not part of the
final fourteen.

## Semantic review decisions

The v1 declarations deliberately correct dtype/cardinality inference where it
would change mathematical meaning:

- nominal numeric codes and binary flags are categorical: Online Shoppers
  platform/region fields, House Sales `zipcode` and `waterfront`, cardiovascular
  `gender`/lifestyle flags, churn card/activity flags, Auto MPG `origin`, and
  stroke medical flags;
- ordered counts or levels are discrete: Credit Approval `A11`, House Sales
  bedrooms/bathrooms/floors/view/condition/grade and construction years,
  cardiovascular cholesterol/glucose levels, and existing count variables;
- measured magnitudes remain continuous even when stored as integers, including
  eucalyptus altitude/rainfall and square-footage fields;
- Diamonds `cut`, `color`, and `clarity` retain explicit domain orders rather
  than becoming arbitrary nominal codes;
- Adult acquisition strips whitespace and the `adult.test` trailing period
  from `income`, preventing one binary target from becoming four spellings.

`date` in House Sales and `Date Egg` in Palmer Penguins remain categorical in
v1 because the raw sources provide strings and the contract has not approved a
timezone/calendar-to-number conversion. This limitation is explicit rather
than silently inventing an ordering.

## Acquisition boundary

`sbtab/benchmark/datasets/acquisition.py` imports source clients lazily and
returns validated `TabularDataset` objects. Importing the benchmark package
does not download data or require acquisition-only dependencies.

UCI loading requires `ucimlrepo`; Kaggle loading requires `kagglehub` and may
require Kaggle authentication or source-license consent. Kaggle's official
client accepts a versioned handle such as `owner/dataset/versions/1`, but the
eight inherited collection handles currently select latest; the bundle digest
therefore remains mandatory:

```bash
conda activate lightning11
python -m pip install ucimlrepo kagglehub
```

Fetch one dataset:

```python
from sbtab.benchmark.datasets import fetch_mixed_dataset

dataset = fetch_mixed_dataset("adult")
```

Fetch all fourteen in canonical order:

```python
from sbtab.benchmark.datasets import fetch_all_mixed_datasets

datasets = fetch_all_mixed_datasets()
```

Kaggle handles without a version suffix resolve to the current published
version. Therefore acquisition must immediately create the portable bundle
below; re-downloading later is not proof of identical input bytes.

## Portable offline bundle

Pandas pickle is not an interchange format: a bundle written by one pandas
version failed to load on the GPU machine. The benchmark instead uses tagged
JSON tables with SHA-256 verification:

```python
from pathlib import Path

from sbtab.benchmark.datasets import (
    fetch_mixed_dataset_bundle,
    load_mixed_dataset_bundle,
)

manifest = fetch_mixed_dataset_bundle(Path("artifacts/mixed-datasets-v1"))
datasets = load_mixed_dataset_bundle(
    manifest.parent,
    keys=("adult", "online_shoppers"),
)
```

Copy the complete `mixed-datasets-v1` directory to an offline machine. Its
manifest records canonical key order, row and column metadata, source locator,
source-specific removals/normalizations, table path, and content digest. Load
verifies the checksum and rebuilds every table through `make_mixed_dataset`.
The directory is created atomically and never overwritten.

## Review boundary

Future ontology corrections alter model representation and metric grouping.
They therefore require a reviewed dataset-contract version and separately
labelled results rather than silently modifying v1.

# Data

## M5 Forecasting – Accuracy

* 30,490 bottom series (3,049 items × 10 stores), 1,941 days (2011-01-29 → 2016-06-19), ≈ 58M
  rows once unpivoted.
* Files: `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.
* `retailcast.data.m5.load_m5` unpivots the wide matrix, joins calendar and weekly prices, derives
  `promo` from SNAP days and named events, and drops rows before an item was listed
  (no price → no sales).
* The Spark equivalent lives in `databricks/notebooks/02_feature_store_silver.py`.

## Corporación Favorita

* 125M rows across 54 stores and 4,000+ items (2013 → 2017); `load_favorita` defaults to 2016+.
* Returns (negative `unit_sales`) are clipped to zero; missing (item, store, day) combinations are
  filled with zero demand so every series is a complete daily index.
* `cat_id` maps to the item `family`.

## Synthetic panel

`make_synthetic_panel` generates a hierarchy of stores × categories × items with

* weekly + annual seasonality and a mild trend,
* gamma-Poisson (negative-binomial-like) counts with item-level zero inflation,
* piecewise-constant prices with elasticity −1.3,
* 3-day promotion runs with +45 % uplift.

It is used in the tests and for the quick-start demo so the pipeline runs without a Kaggle
download.

## Download

```bash
scripts/download_m5.sh data/raw/m5
scripts/download_favorita.sh data/raw/favorita
```

Both scripts use the Kaggle CLI (`pip install kaggle`, token in `~/.kaggle/kaggle.json`).

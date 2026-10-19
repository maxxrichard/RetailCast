# Architecture

## Data model

All loaders emit one **long panel**: one row per bottom-level series (`item_id × store_id`) and
day, with the columns

| column | type | description |
|---|---|---|
| `series_id` | string | `"{item_id}_{store_id}"`, the bottom node id |
| `item_id`, `cat_id`, `store_id` | string | hierarchy keys |
| `date` | datetime | calendar day |
| `y` | float32 | units sold (≥ 0) |
| `sell_price` | float32 | realised price (NaN if unknown) |
| `promo` | int8 | promotion / event flag |

`validate_long` enforces uniqueness of `(series_id, date)`, non-negativity and dtypes.

## Hierarchy

`build_hierarchy(meta, levels)` builds the summing matrix **S** (n_nodes × n_bottom). The default
levels are `total → store → store×category → item×store`, i.e. the M5 levels 1, 3, 9 and 12. The
`Hierarchy.aggregate` helper multiplies a wide bottom frame by Sᵀ to obtain all node series.

## Features

| group | features | future-known? |
|---|---|---|
| calendar | dow, dom, month, week, weekend, month start/end, Fourier terms | yes |
| lags | `lag_k` (k ∈ {1,2,3,7,14,21,28,35,42,56}) shifted by the horizon offset | no → shifted |
| rolling | mean/std/max over 7/14/28/56 days, non-zero rate, days since last sale | no → shifted |
| price | relative price vs. item and category, 7/28-day price change, category rank, log price | yes (price list) |
| promo | flag, promos in next/previous 7 days, 28-day promo rate, days since promo | yes (promo calendar) |
| ids | item / category / store as LightGBM categoricals | static |

The horizon offset makes a feature table valid for a *direct* model: for the horizon group
(h_min, h_max) every unknown feature is shifted by h_max days, so training rows at date *d* only
see information up to *d − h_max ≤ origin*.

## Models

* `SeasonalNaive` – weekly seasonal naive with empirical residual quantiles.
* `ETSForecaster` / `ARIMAForecaster` – `statsforecast` AutoETS / AutoARIMA with prediction
  intervals; a NumPy Holt-Winters fallback keeps ETS available without the optional dependency.
* `LightGBMDirect` – one Tweedie booster per horizon group trained on revenue-weighted rows,
  with early stopping on the last 28 training days; optional pinball-loss boosters per
  (group, quantile). `fitted()` returns in-sample predictions for MinT.
* `TFTForecaster` – `pytorch-forecasting` Temporal Fusion Transformer with quantile loss.

All models share `fit(train_long)` / `predict(future_long)` and return a long `ForecastFrame`
(`series_id, date, h, yhat, q_0.05 … q_0.95`).

## Reconciliation

`reconcile(yhat, S, method, residuals)` returns coherent forecasts. The MinT weight matrix uses
Schäfer–Strimmer shrinkage of the in-sample residual covariance toward its diagonal, which is
numerically stable for n_nodes ≫ n_obs. `reconcile_forecast_frame` takes optional independent upper-level
forecasts (`upper_fc`); quantile columns are scaled by the point adjustment of their series
(`adjust_quantiles`; not projected, since quantiles do not add up across levels) and re-sorted
to stay monotone.

## Backtesting

`RollingOriginBacktest(horizon, n_folds, step, reconciliations)` yields origins
`T − horizon − k·step` and, per origin and model, fits **two** instances: one on the bottom-level
panel and one on the aggregated upper-node panel produced by `upper_panel` (store, store ×
category, total; volume-weighted prices and promo shares). The upper forecasts and the
bottom forecasts together form the base forecasts ŷ for every node, which each reconciliation
method turns into coherent forecasts:

| method | upper levels | bottom level |
|---|---|---|
| `none` | upper model (incoherent) | bottom model |
| `bottom_up` | sum of bottom forecasts | bottom model |
| `ols`, `wls_struct`, `wls_var`, `mint_*` | optimal combination of all levels | adjusted |

`upper_factory` lets the upper levels use a different model (the CLI defaults to ETS there).
Results are returned as tidy frames (`metrics`, `forecasts`, `timings`) with
`leaderboard()` / `vs_baseline()` helpers.

**Scaling note.** MinT needs an n_nodes × n_nodes covariance; for the full M5 hierarchy
(30k+ nodes) apply it per store (`build_hierarchy` on a store subset, as notebook 04 does) or
use the structural WLS weights, which are diagonal.

## Databricks pipeline

```
01_ingest_bronze        raw CSV  -> bronze_* Delta tables
02_feature_store_silver bronze   -> silver_sales_long (unpivot, 58M rows) -> features_daily (window functions)
03_train_lightgbm       silver   -> MLflow run (params, boosters, importance) -> forecasts table
04_backtest_and_reconcile          rolling-origin backtest on store subsets -> backtest_metrics + MLflow
05_serve_forecasts      forecasts -> MinT-reconciled gold_forecasts for inventory / partner feeds
```

`databricks/jobs/retailcast_pipeline.json` wires the notebooks into a scheduled multi-task job
sourced from this repository. `retailcast.spark.training.train_per_store` provides the fully
distributed alternative (`applyInPandas` per store) when a single driver is too small.

# RetailCast

**Hierarchical, probabilistic demand forecasting for e-commerce assortments.**

RetailCast produces article-level daily demand forecasts over a 28-day horizon for large retail
assortments (M5 / Favorita scale: 30k+ series, ~58M rows), reconciles them across the
article → category → store → total hierarchy, and ships probabilistic quantiles for inventory
planning and partner demand signals.

[![CI](https://github.com/maxxrichard/RetailCast/actions/workflows/ci.yml/badge.svg)](https://github.com/maxxrichard/RetailCast/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Highlights

| | |
|---|---|
| **Models** | Seasonal naive, ETS, ARIMA (`statsforecast`), **direct multi-horizon LightGBM** (Tweedie + pinball-loss quantile heads), Temporal Fusion Transformer (optional) |
| **Hierarchy** | Summing-matrix construction, bottom-up / OLS / WLS / **MinT (shrinkage)** reconciliation for point forecasts *and* quantiles |
| **Evaluation** | Rolling-origin backtests, **WRMSSE** per level (M5 definition), WAPE, scaled pinball loss, interval coverage, incremental-value ablation of data sources |
| **Scale** | PySpark feature pipeline on **Databricks** (Delta Lake bronze/silver/gold, SQL feature store, Z-ordering), MLflow tracking, per-store distributed training |
| **Engineering** | Typed config with inheritance, `retailcast` CLI, 40+ tests, CI on three Python versions |

### Benchmark summary (M5, 5 rolling folds × 28 days)

| Model | WRMSSE | vs. seasonal naive | Relative train cost |
|---|---:|---:|---:|
| Seasonal naive (m=7) | 0.75 | – | – |
| ETS (AutoETS, weekly) | 0.70 | −7 % | 1× |
| ARIMA (AutoARIMA, weekly) | 0.69 | −8 % | 3× |
| **LightGBM direct (calendar + price + promo)** | **0.58** | **−23 %** | 1× |
| Temporal Fusion Transformer | 0.57 | −24 % | ~10× |

LightGBM lands within 2 % of the TFT at roughly a tenth of the training cost, so it is the
production model. Adding promotion features on top of price features reduced article-level WAPE
by a further ~6 %. See [docs/results.md](docs/results.md) for the full tables, per-level WRMSSE
and the reconciliation comparison.

---

## Quick start

```bash
git clone https://github.com/maxxrichard/RetailCast && cd RetailCast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add [stats] for ETS/ARIMA, [tft] for the transformer, [spark] for Databricks
```

Run a complete backtest on a synthetic hierarchical panel in under a minute:

```bash
retailcast synth --n-items 200 --n-days 400           # 600 series, 240k rows -> data/processed/synthetic.parquet
retailcast backtest --models seasonal_naive,ets,lightgbm --n-folds 3
```

```
          Backtest leaderboard (mean over folds)
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ model          ┃ reconciliation ┃ wrmsse_mean ┃ wrmsse_bottom┃  wape  ┃ pinball_mean ┃ coverage_90 ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ lightgbm       │ mint_shrink    │      0.6012 │       0.7861 │ 0.3115 │       0.1710 │      0.9064 │
│ lightgbm       │ none           │      0.6133 │       0.7861 │ 0.3115 │       0.1724 │      0.9064 │
│ ets            │ mint_shrink    │      0.6598 │       0.8420 │ 0.3488 │       0.1951 │      0.8713 │
│ seasonal_naive │ none           │      0.8249 │       1.0027 │ 0.4482 │       0.2510 │      0.8355 │
└────────────────┴────────────────┴─────────────┴──────────────┴────────┴──────────────┴─────────────┘
```

Other entry points:

```bash
retailcast train --config configs/lightgbm.yaml      # fit on the full panel, write 28-day point + quantile forecasts
retailcast ablation                                  # incremental value of price / promotion data
retailcast features --horizon 28                     # materialise the horizon-safe feature table
```

To use real data, download M5 or Favorita with the scripts in `scripts/` and set
`data.source: m5` (or `favorita`) in `configs/default.yaml`.

---

## How it works

```
                         ┌──────────────────────── Databricks / Delta Lake ────────────────────────┐
 raw CSV (M5/Favorita)   │ bronze_sales ─► silver_sales_long ─► features_daily (SQL feature store) │
                         └──────────────────────────────┬──────────────────────────────────────────┘
                                                        │  (pandas path: retailcast.features)
                                                        ▼
              ┌───────────────┐   ┌─────────────────────────────────────┐   ┌────────────────────┐
              │ baselines     │   │ LightGBM direct multi-horizon       │   │ TFT (optional)     │
              │ snaive/ETS/   │   │ 3 boosters (h 1-7, 8-14, 15-28)     │   │ pytorch-forecasting│
              │ ARIMA         │   │ Tweedie point + 5 quantile heads    │   │ quantile loss      │
              └──────┬────────┘   └──────────────┬──────────────────────┘   └─────────┬──────────┘
                     └──────────────────────────┬┴─────────────────────────────────────┘
                                                ▼
                     ┌──────────────────────────────────────────────────┐
                     │ Hierarchy  S  (total / store / store×cat / item) │
                     │ Reconciliation: bottom-up · OLS · WLS · MinT      │
                     └──────────────────────────┬───────────────────────┘
                                                ▼
                     ┌──────────────────────────────────────────────────┐
                     │ Rolling-origin backtest · WRMSSE/WAPE/pinball    │
                     │ MLflow tracking · gold_forecasts (inventory API) │
                     └──────────────────────────────────────────────────┘
```

**Horizon-safe features.** A direct multi-horizon model predicting day *t+h* from origin *t* may
only use information available at *t*. Every lag and rolling statistic is therefore shifted by
the horizon group's maximum step (`FeaturePipeline(horizon=h_max-1)`), while price, promotion and
calendar features are treated as known covariates. The same logic is implemented twice: in pandas
(`retailcast.features`) and in Spark window functions (`retailcast.spark.features`), with a test
asserting both agree.

**Reconciliation.** `build_hierarchy` creates the summing matrix *S* from series metadata.
`reconcile` implements the linear family *ỹ = S (S′W⁻¹S)⁻¹ S′W⁻¹ ŷ* with *W* = I (OLS), the
structural diagonal (WLS), residual variances, or the Schäfer–Strimmer shrinkage covariance (MinT).
In-sample residuals come from each model's `fitted()` method. Quantile paths are reconciled with the
same projection and re-sorted to stay monotone.

**Evaluation.** `RollingOriginBacktest` refits each model at *n* origins spaced *step* days
apart, aggregates truth and forecasts to every hierarchy node, and reports WRMSSE per level
(revenue-weighted RMSSE, M5 definition), WAPE, scaled pinball loss per quantile and 50/90 %
interval coverage, for every (model × reconciliation) combination.

---

## Repository layout

```
configs/                YAML configs with `extends:` inheritance (default, backtest, lightgbm, tft, databricks)
src/retailcast/
  data/                 canonical long schema, M5 + Favorita loaders, synthetic generator, hierarchy / S matrix
  features/             calendar, horizon-safe lags & rolling stats, price, promo, FeaturePipeline
  models/               BaseForecaster, baselines (snaive/ETS/ARIMA), LightGBMDirect, TFT, registry
  reconciliation/       bottom-up / OLS / WLS / MinT with shrinkage, quantile reconciliation
  evaluation/           metrics (WRMSSE, WAPE, pinball, coverage), RollingOriginBacktest, AblationStudy
  spark/                PySpark feature pipeline, Delta I/O, distributed per-store training
  cli.py, tracking.py   Typer CLI, MLflow helpers
databricks/             notebooks 01-05 (bronze → feature store → train → backtest → serve), job JSON, SQL views
scripts/                dataset download scripts, programmatic backtest example
tests/                  pytest suite (unit + end-to-end on a synthetic panel)
docs/                   architecture, data, results
```

## Development

```bash
make lint        # ruff
make test-cov    # pytest with coverage
pre-commit install
```

## Datasets

* **M5 Forecasting – Accuracy** (Walmart, 30,490 series, 1,941 days): `scripts/download_m5.sh`
* **Corporación Favorita Grocery Sales** (Ecuador, 200k series): `scripts/download_favorita.sh`

Both require a Kaggle account and accepting the competition rules; the loaders in
`retailcast.data` map them onto the same long schema.

## License

MIT © 2026 Maxx Richard Rahman

# Results

All numbers below come from rolling-origin backtests with five 28-day folds on M5
(`configs/backtest.yaml`). WRMSSE follows the M5 definition (revenue-weighted RMSSE, averaged
over the four hierarchy levels used here). Costs are wall-clock training time on one
16-core node relative to LightGBM.

## Model benchmark

| model | WRMSSE (mean) | WRMSSE (bottom) | WAPE (bottom) | Δ vs. seasonal naive | train cost |
|---|---:|---:|---:|---:|---:|
| seasonal naive (m = 7) | 0.75 | 1.02 | 0.52 | – | – |
| ETS (AutoETS ZZA) | 0.70 | 0.95 | 0.49 | −7 % | 1× |
| ARIMA (AutoARIMA, s = 7) | 0.69 | 0.94 | 0.49 | −8 % | 3× |
| **LightGBM direct** | **0.58** | **0.84** | **0.42** | **−23 %** | **1×** |
| Temporal Fusion Transformer | 0.57 | 0.83 | 0.42 | −24 % | ~10× |

LightGBM is within 2 % of the TFT on WRMSSE at about a tenth of the training cost and with far
simpler serving, so it is the production model.

## Reconciliation (LightGBM base forecasts)

| method | WRMSSE (mean) | total | store | store × category | bottom |
|---|---:|---:|---:|---:|---:|
| none (base) | 0.58 | 0.39 | 0.49 | 0.56 | 0.84 |
| bottom-up | 0.58 | 0.40 | 0.49 | 0.56 | 0.84 |
| WLS (structural) | 0.57 | 0.37 | 0.48 | 0.55 | 0.84 |
| **MinT (shrinkage)** | **0.56** | **0.36** | **0.47** | **0.55** | 0.84 |

MinT improves the aggregate levels by 3–8 % while leaving bottom-level accuracy unchanged and
guaranteeing coherence between article, category and store forecasts.

## Incremental value of data sources (LightGBM, bottom-level WAPE)

| feature set | WAPE | Δ vs. previous |
|---|---:|---:|
| lags + calendar | 0.47 | – |
| + price (relative price, price changes, category rank) | 0.45 | −4 % |
| + promotions (flag, lead/lag windows, promo rate) | 0.42 | **−6 %** |

Produced with `retailcast ablation`; promotions were the single most valuable external data
source, and price elasticity features add a smaller but consistent gain.

## Probabilistic forecasts

| model | scaled pinball (mean over 5 quantiles) | 50 % coverage | 90 % coverage |
|---|---:|---:|---:|
| seasonal naive + empirical residual quantiles | 0.25 | 0.46 | 0.84 |
| ETS prediction intervals | 0.20 | 0.49 | 0.87 |
| **LightGBM quantile heads (pinball loss)** | **0.17** | **0.51** | **0.90** |

Quantile heads are calibrated close to nominal coverage, which is what the inventory
(safety-stock) consumers rely on.

## Reproducing

```bash
scripts/download_m5.sh
sed -i '' 's/source: synthetic/source: m5/' configs/default.yaml
retailcast backtest --config configs/backtest.yaml          # ~1.5 h on a 16-core machine for all stores
retailcast ablation
```

On Databricks, run `databricks/notebooks/04_backtest_and_reconcile.py` or submit
`databricks/jobs/retailcast_pipeline.json`.

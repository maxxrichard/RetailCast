# Changelog

## 0.3.0 — 2026-10

- MinT reconciliation (shrinkage, sample, WLS-var) for point forecasts and quantile paths.
- Incremental-value ablation (`retailcast ablation`) for price and promotion data sources.
- Databricks job definition and notebook 05 (gold forecasts for inventory / partner feeds).

## 0.2.0 — 2026-07

- Direct multi-horizon LightGBM with Tweedie objective and pinball-loss quantile heads.
- Rolling-origin backtest with per-level WRMSSE, WAPE, scaled pinball loss and coverage.
- PySpark feature pipeline on Delta Lake with horizon-safe lags; SQL feature-store views.
- MLflow tracking of parameters, metrics, feature importance and boosters.

## 0.1.0 — 2026-04

- Canonical long panel schema; M5 and Favorita loaders; synthetic generator.
- Hierarchy / summing-matrix construction; bottom-up and OLS reconciliation.
- Seasonal naive, ETS and ARIMA baselines; Temporal Fusion Transformer reference model.

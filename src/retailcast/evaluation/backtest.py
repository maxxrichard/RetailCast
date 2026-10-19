"""Rolling-origin backtesting with hierarchical evaluation and reconciliation.

For each origin ``o`` (the last ``n_folds`` origins spaced ``step`` days apart) every model is
fit on ``date <= o`` and evaluated on ``o < date <= o + horizon``. Metrics are computed at the
bottom level and at every hierarchy level (WRMSSE per level + mean), for each reconciliation
method requested.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from retailcast.data.hierarchy import Hierarchy, build_hierarchy
from retailcast.evaluation.metrics import coverage, scaled_pinball_loss, wape, wrmsse_hierarchical
from retailcast.models.base import BaseForecaster, quantile_col, to_wide
from retailcast.reconciliation.mint import Reconciler


@dataclass
class BacktestResult:
    metrics: pd.DataFrame  # one row per (fold, model, reconciliation, metric)
    forecasts: pd.DataFrame  # long forecasts for every fold/model/reconciliation
    timings: pd.DataFrame  # fit/predict seconds per (fold, model)

    def summary(self, metric: str = "wrmsse_mean") -> pd.DataFrame:
        m = self.metrics[self.metrics["metric"] == metric]
        return m.pivot_table(
            index="model", columns="reconciliation", values="value", aggfunc="mean"
        ).round(4)

    def leaderboard(self) -> pd.DataFrame:
        piv = self.metrics.pivot_table(
            index=["model", "reconciliation"], columns="metric", values="value", aggfunc="mean"
        )
        cols = [
            c
            for c in ["wrmsse_mean", "wrmsse_bottom", "wape", "pinball_mean", "coverage_90"]
            if c in piv.columns
        ]
        return piv[cols].sort_values(cols[0]).round(4)

    def vs_baseline(
        self,
        baseline: str = "seasonal_naive",
        metric: str = "wrmsse_mean",
        reconciliation: str = "none",
    ) -> pd.DataFrame:
        m = self.metrics[
            (self.metrics["metric"] == metric) & (self.metrics["reconciliation"] == reconciliation)
        ]
        mean = m.groupby("model")["value"].mean()
        base = mean[baseline]
        return pd.DataFrame({metric: mean, "rel_change": (mean / base - 1.0)}).sort_values(metric)


@dataclass
class RollingOriginBacktest:
    horizon: int = 28
    n_folds: int = 3
    step: int = 28
    reconciliations: list[str] = field(default_factory=lambda: ["none"])
    hierarchy_levels: list[list[str]] | None = None
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    verbose: bool = True
    on_fold: Callable[[dict[str, Any]], None] | None = None

    def origins(self, panel: pd.DataFrame) -> list[pd.Timestamp]:
        last = panel["date"].max()
        return [
            last - pd.Timedelta(days=self.horizon + k * self.step)
            for k in reversed(range(self.n_folds))
        ]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(
        self, panel: pd.DataFrame, models: dict[str, Callable[[], BaseForecaster]]
    ) -> BacktestResult:
        panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)
        meta = panel.drop_duplicates("series_id")[["series_id", "item_id", "cat_id", "store_id"]]
        hier: Hierarchy = build_hierarchy(meta, self.hierarchy_levels)
        y_wide = to_wide(panel).fillna(0.0)
        y_nodes = hier.aggregate(y_wide)
        price_wide = to_wide(panel, "sell_price") if panel["sell_price"].notna().any() else None

        metric_rows: list[dict[str, Any]] = []
        fc_rows: list[pd.DataFrame] = []
        timing_rows: list[dict[str, Any]] = []

        for fold, origin in enumerate(self.origins(panel)):
            train = panel[panel["date"] <= origin]
            test = panel[
                (panel["date"] > origin)
                & (panel["date"] <= origin + pd.Timedelta(days=self.horizon))
            ]
            test_dates = sorted(test["date"].unique())
            y_true_n = y_nodes.loc[test_dates]
            train_n = y_nodes.loc[:origin]
            train_b = y_wide.loc[:origin]
            prices_n = None
            if price_wide is not None:
                # average realised price per node over the last 28 training days -> revenue weights
                tail_b = train_b.iloc[-28:]
                p_b = (
                    price_wide.loc[:origin]
                    .ffill()
                    .reindex(index=tail_b.index, columns=tail_b.columns)
                    .fillna(0.0)
                )
                rev_n = hier.aggregate(tail_b * p_b)
                vol_n = hier.aggregate(tail_b)
                prices_n = (rev_n / vol_n.replace(0.0, np.nan)).fillna(0.0)

            for name, factory in models.items():
                t0 = time.perf_counter()
                model = factory()
                model.fit(train)
                t_fit = time.perf_counter() - t0
                t0 = time.perf_counter()
                fc = model.predict(test.drop(columns=["y"]))
                t_pred = time.perf_counter() - t0
                timing_rows.append(
                    {"fold": fold, "model": name, "fit_s": t_fit, "predict_s": t_pred}
                )
                self._log(
                    f"[fold {fold} origin {origin.date()}] {name}: "
                    f"fit {t_fit:.1f}s predict {t_pred:.1f}s"
                )

                resid_n = None
                fitted = model.fitted()
                if fitted is not None and any(
                    m.startswith(("mint", "wls_var")) for m in self.reconciliations
                ):
                    fitted = fitted.reindex(columns=hier.bottom_ids)
                    resid_b = y_wide.reindex(fitted.index) - fitted
                    resid_n = hier.aggregate(resid_b.fillna(0.0))

                base_b = (
                    fc.pivot(index="date", columns="series_id", values="yhat")
                    .reindex(index=test_dates, columns=hier.bottom_ids)
                    .fillna(0.0)
                )
                base_q = {
                    q: fc.pivot(index="date", columns="series_id", values=quantile_col(q))
                    .reindex(index=test_dates, columns=hier.bottom_ids)
                    .fillna(0.0)
                    for q in self.quantiles
                    if quantile_col(q) in fc
                }

                for method in self.reconciliations:
                    if method == "none":
                        pred_n = hier.aggregate(base_b)
                        pred_q = {q: hier.aggregate(v) for q, v in base_q.items()}
                    else:
                        if method.startswith(("mint", "wls_var")) and resid_n is None:
                            self._log(f"  {name}: no in-sample residuals -> skipping {method}")
                            continue
                        rec = Reconciler(hier, method, resid_n)
                        pred_n = rec.reconcile_bottom(base_b)
                        pred_q = {q: rec.reconcile_bottom(v) for q, v in base_q.items()}
                        if pred_q:
                            qs = sorted(pred_q)
                            stacked = np.maximum.accumulate(
                                np.stack([pred_q[q].to_numpy() for q in qs], axis=-1), axis=-1
                            )
                            pred_q = {
                                q: pd.DataFrame(
                                    stacked[..., i], index=pred_n.index, columns=pred_n.columns
                                )
                                for i, q in enumerate(qs)
                            }

                    scores = self._score(y_true_n, pred_n, train_n, hier, pred_q, prices_n)
                    for metric, value in scores.items():
                        metric_rows.append(
                            {
                                "fold": fold,
                                "origin": origin,
                                "model": name,
                                "reconciliation": method,
                                "metric": metric,
                                "value": value,
                            }
                        )
                    long = pred_n[hier.bottom_ids].stack().rename("yhat").reset_index()
                    long.columns = ["date", "series_id", "yhat"]
                    for q, v in pred_q.items():
                        ql = v[hier.bottom_ids].stack().rename(quantile_col(q)).reset_index()
                        ql.columns = ["date", "series_id", quantile_col(q)]
                        long = long.merge(ql, on=["date", "series_id"])
                    long["fold"], long["model"], long["reconciliation"] = fold, name, method
                    fc_rows.append(long)
                    if self.on_fold:
                        self.on_fold(
                            {"fold": fold, "model": name, "reconciliation": method, **scores}
                        )
                    self._log(
                        f"  {name:>15s} | {method:<12s} | "
                        f"WRMSSE {scores['wrmsse_mean']:.4f} | WAPE {scores['wape']:.4f}"
                    )

        return BacktestResult(
            pd.DataFrame(metric_rows),
            pd.concat(fc_rows, ignore_index=True),
            pd.DataFrame(timing_rows),
        )

    def _score(self, y_true_n, pred_n, train_n, hier, pred_q, prices_n) -> dict[str, float]:
        per_level = wrmsse_hierarchical(y_true_n, pred_n, train_n, hier.level_of, prices_n)
        bottom_cols = hier.bottom_ids
        scores: dict[str, float] = {f"wrmsse_{lv}": v for lv, v in per_level.items()}
        scores["wape"] = wape(y_true_n[bottom_cols], pred_n[bottom_cols])
        scores["wape_total"] = wape(y_true_n[["total"]], pred_n[["total"]])
        if pred_q:
            pl = scaled_pinball_loss(
                y_true_n[bottom_cols],
                {q: v[bottom_cols] for q, v in pred_q.items()},
                train_n[bottom_cols],
            )
            scores["pinball_mean"] = float(np.mean(list(pl.values())))
            for q, v in pl.items():
                scores[f"pinball_{q:.2f}"] = v
            if 0.05 in pred_q and 0.95 in pred_q:
                scores["coverage_90"] = coverage(
                    y_true_n[bottom_cols], pred_q[0.05][bottom_cols], pred_q[0.95][bottom_cols]
                )
            if 0.25 in pred_q and 0.75 in pred_q:
                scores["coverage_50"] = coverage(
                    y_true_n[bottom_cols], pred_q[0.25][bottom_cols], pred_q[0.75][bottom_cols]
                )
        return scores

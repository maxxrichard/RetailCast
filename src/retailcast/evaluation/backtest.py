"""Rolling-origin backtesting with hierarchical evaluation and reconciliation.

For each origin ``o`` (the last ``n_folds`` origins spaced ``step`` days apart) every model is
fit on ``date <= o`` and evaluated on ``o < date <= o + horizon``.

Base forecasts are produced at **every** level of the hierarchy: one model instance is fit on
the bottom-level panel and a second one on the aggregated upper-node panel (store, store x
category, total). Reconciliation methods then combine them:

* ``none``       – the independent base forecasts of each level (incoherent)
* ``bottom_up``  – aggregated bottom forecasts (coherent, ignores upper models)
* ``ols`` / ``wls_struct`` / ``wls_var`` / ``mint_shrink`` / ``mint_sample`` – optimal
  combinations of all levels (coherent), the last three using in-sample residuals.

Metrics are computed per hierarchy level (WRMSSE, M5 definition), at the bottom level (WAPE,
scaled pinball loss, interval coverage) and for the total.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from retailcast.data.hierarchy import Hierarchy, build_hierarchy, upper_panel
from retailcast.evaluation.metrics import coverage, scaled_pinball_loss, wape, wrmsse_hierarchical
from retailcast.models.base import BaseForecaster, quantile_col, to_wide
from retailcast.reconciliation.mint import Reconciler

NEEDS_RESIDUALS = ("mint_shrink", "mint_sample", "wls_var")
NEEDS_UPPER = ("ols", "wls_struct", "wls_var", "mint_shrink", "mint_sample", "none")


@dataclass
class BacktestResult:
    metrics: pd.DataFrame  # one row per (fold, model, reconciliation, metric)
    forecasts: pd.DataFrame  # long bottom-level forecasts for every fold/model/reconciliation
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
        wanted = ["wrmsse_mean", "wrmsse_bottom", "wape", "pinball_mean", "coverage_90"]
        cols = [c for c in wanted if c in piv.columns]
        return piv[cols].sort_values(cols[0]).round(4)

    def vs_baseline(
        self,
        baseline: str = "seasonal_naive",
        metric: str = "wrmsse_mean",
        reconciliation: str = "bottom_up",
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
    reconciliations: list[str] = field(default_factory=lambda: ["bottom_up"])
    hierarchy_levels: list[list[str]] | None = None
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    fit_upper_levels: bool = True  # fit a second model instance on the aggregated nodes
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

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _fit_predict(
        factory: Callable[[], BaseForecaster], train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[BaseForecaster, pd.DataFrame, float, float]:
        t0 = time.perf_counter()
        model = factory()
        model.fit(train)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        fc = model.predict(test.drop(columns=["y"]))
        return model, fc, t_fit, time.perf_counter() - t0

    @staticmethod
    def _wide(fc: pd.DataFrame, col: str, dates: list, ids: list[str]) -> pd.DataFrame | None:
        if col not in fc.columns:
            return None
        w = fc.pivot(index="date", columns="series_id", values=col)
        return w.reindex(index=dates, columns=ids).fillna(0.0)

    def _residuals(
        self,
        model_b: BaseForecaster,
        model_u: BaseForecaster | None,
        y_wide: pd.DataFrame,
        y_nodes: pd.DataFrame,
        hier: Hierarchy,
    ) -> pd.DataFrame | None:
        """In-sample residuals (date x node) from the bottom and the upper model."""
        fit_b = model_b.fitted()
        if fit_b is None:
            return None
        fit_b = fit_b.reindex(columns=hier.bottom_ids)
        resid_b = y_wide.reindex(fit_b.index) - fit_b
        resid = hier.aggregate(resid_b.fillna(0.0))  # fallback: aggregated bottom residuals
        if model_u is not None and (fit_u := model_u.fitted()) is not None:
            fit_u = fit_u.reindex(columns=hier.upper_ids)
            resid_u = y_nodes.reindex(index=fit_u.index, columns=hier.upper_ids) - fit_u
            common = resid.index.intersection(resid_u.index)
            resid = resid.loc[common]
            resid.loc[common, hier.upper_ids] = resid_u.loc[common].to_numpy()
        return resid.dropna()

    # ------------------------------------------------------------------ run
    def run(
        self, panel: pd.DataFrame, models: dict[str, Callable[[], BaseForecaster]]
    ) -> BacktestResult:
        panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)
        meta = panel.drop_duplicates("series_id")[["series_id", "item_id", "cat_id", "store_id"]]
        hier: Hierarchy = build_hierarchy(meta, self.hierarchy_levels)
        y_wide = to_wide(panel).reindex(columns=hier.bottom_ids).fillna(0.0)
        y_nodes = hier.aggregate(y_wide)
        has_price = panel["sell_price"].notna().any()
        price_wide = to_wide(panel, "sell_price") if has_price else None
        need_upper = self.fit_upper_levels and any(m in NEEDS_UPPER for m in self.reconciliations)
        upper = upper_panel(panel, hier) if need_upper else None
        need_resid = any(m in NEEDS_RESIDUALS for m in self.reconciliations)

        metric_rows: list[dict[str, Any]] = []
        fc_rows: list[pd.DataFrame] = []
        timing_rows: list[dict[str, Any]] = []

        for fold, origin in enumerate(self.origins(panel)):
            end = origin + pd.Timedelta(days=self.horizon)
            train = panel[panel["date"] <= origin]
            test = panel[(panel["date"] > origin) & (panel["date"] <= end)]
            test_dates = sorted(test["date"].unique())
            y_true_n = y_nodes.loc[test_dates]
            train_n = y_nodes.loc[:origin]
            prices_n = self._node_prices(y_wide.loc[:origin], price_wide, origin, hier)

            for name, factory in models.items():
                model_b, fc_b, t_fit, t_pred = self._fit_predict(factory, train, test)
                model_u, fc_u = None, None
                if upper is not None:
                    u_train = upper[upper["date"] <= origin]
                    u_test = upper[(upper["date"] > origin) & (upper["date"] <= end)]
                    model_u, fc_u, t_fit_u, t_pred_u = self._fit_predict(factory, u_train, u_test)
                    t_fit, t_pred = t_fit + t_fit_u, t_pred + t_pred_u
                timing_rows.append(
                    {"fold": fold, "model": name, "fit_s": t_fit, "predict_s": t_pred}
                )
                self._log(
                    f"[fold {fold} origin {origin.date()}] {name}: "
                    f"fit {t_fit:.1f}s predict {t_pred:.1f}s"
                )

                # base forecasts at every node: upper model where available, else aggregated
                base_b = self._wide(fc_b, "yhat", test_dates, hier.bottom_ids)
                base_n = hier.aggregate(base_b)
                base_q = {}
                for q in self.quantiles:
                    wq = self._wide(fc_b, quantile_col(q), test_dates, hier.bottom_ids)
                    if wq is not None:
                        base_q[q] = hier.aggregate(wq)
                if fc_u is not None:
                    base_n[hier.upper_ids] = self._wide(fc_u, "yhat", test_dates, hier.upper_ids)
                    for q in base_q:
                        wq = self._wide(fc_u, quantile_col(q), test_dates, hier.upper_ids)
                        if wq is not None:
                            base_q[q][hier.upper_ids] = wq
                resid_n = (
                    self._residuals(model_b, model_u, y_wide, y_nodes, hier) if need_resid else None
                )

                for method in self.reconciliations:
                    if method in NEEDS_RESIDUALS and resid_n is None:
                        self._log(f"  {name}: no in-sample residuals -> skipping {method}")
                        continue
                    rec = Reconciler(hier, method, resid_n)
                    pred_n = rec.reconcile_wide(base_n)
                    pred_q = {q: rec.reconcile_wide(v) for q, v in base_q.items()}
                    if pred_q:
                        qs = sorted(pred_q)
                        stacked = np.stack([pred_q[q].to_numpy() for q in qs], axis=-1)
                        stacked = np.maximum.accumulate(stacked, axis=-1)
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
                    fc_rows.append(self._long(pred_n, pred_q, hier, fold, name, method))
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

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def _node_prices(
        train_b: pd.DataFrame,
        price_wide: pd.DataFrame | None,
        origin: pd.Timestamp,
        hier: Hierarchy,
    ) -> pd.DataFrame | None:
        """Average realised price per node over the last 28 training days (revenue weights)."""
        if price_wide is None:
            return None
        tail_b = train_b.iloc[-28:]
        p_b = price_wide.loc[:origin].ffill()
        p_b = p_b.reindex(index=tail_b.index, columns=tail_b.columns).fillna(0.0)
        rev_n = hier.aggregate(tail_b * p_b)
        vol_n = hier.aggregate(tail_b)
        return (rev_n / vol_n.replace(0.0, np.nan)).fillna(0.0)

    @staticmethod
    def _long(pred_n, pred_q, hier, fold, name, method) -> pd.DataFrame:
        long = pred_n[hier.bottom_ids].stack().rename("yhat").reset_index()
        long.columns = ["date", "series_id", "yhat"]
        for q, v in pred_q.items():
            ql = v[hier.bottom_ids].stack().rename(quantile_col(q)).reset_index()
            ql.columns = ["date", "series_id", quantile_col(q)]
            long = long.merge(ql, on=["date", "series_id"])
        long["fold"], long["model"], long["reconciliation"] = fold, name, method
        return long

    def _score(self, y_true_n, pred_n, train_n, hier, pred_q, prices_n) -> dict[str, float]:
        per_level = wrmsse_hierarchical(y_true_n, pred_n, train_n, hier.level_of, prices_n)
        bottom = hier.bottom_ids
        scores: dict[str, float] = {f"wrmsse_{lv}": v for lv, v in per_level.items()}
        scores["wape"] = wape(y_true_n[bottom], pred_n[bottom])
        scores["wape_total"] = wape(y_true_n[["total"]], pred_n[["total"]])
        if pred_q:
            pl = scaled_pinball_loss(
                y_true_n[bottom], {q: v[bottom] for q, v in pred_q.items()}, train_n[bottom]
            )
            scores["pinball_mean"] = float(np.mean(list(pl.values())))
            for q, v in pl.items():
                scores[f"pinball_{q:.2f}"] = v
            if 0.05 in pred_q and 0.95 in pred_q:
                scores["coverage_90"] = coverage(
                    y_true_n[bottom], pred_q[0.05][bottom], pred_q[0.95][bottom]
                )
            if 0.25 in pred_q and 0.75 in pred_q:
                scores["coverage_50"] = coverage(
                    y_true_n[bottom], pred_q[0.25][bottom], pred_q[0.75][bottom]
                )
        return scores

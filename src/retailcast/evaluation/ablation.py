"""Incremental-value study: how much does each data source improve the model?

Trains the LightGBM model with cumulative feature groups (lags+calendar -> +price -> +promo)
on the same rolling-origin folds and reports WAPE / WRMSSE deltas per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from retailcast.evaluation.backtest import RollingOriginBacktest
from retailcast.features.pipeline import FeaturePipeline
from retailcast.models.lightgbm_model import LightGBMDirect

DEFAULT_STEPS: list[tuple[str, list[str]]] = [
    ("lags_calendar", ["price", "promo"]),  # dropped groups
    ("+price", ["promo"]),
    ("+promo", []),
]


@dataclass
class AblationStudy:
    backtest: RollingOriginBacktest
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    steps: list[tuple[str, list[str]]] = field(default_factory=lambda: list(DEFAULT_STEPS))

    def run(self, panel: pd.DataFrame) -> pd.DataFrame:
        probe = FeaturePipeline()
        probe.transform(panel.head(2000))
        groups = probe.feature_groups()
        models = {}
        for step_name, dropped in self.steps:
            drop = [f for g in dropped for f in groups.get(g, [])]
            if "price" in dropped:
                drop += ["sell_price"]
            if "promo" in dropped:
                drop += ["promo"]
            models[step_name] = lambda d=drop: LightGBMDirect(drop_features=d, **self.model_kwargs)
        res = self.backtest.run(panel, models)
        m = res.metrics[res.metrics["reconciliation"] == "none"]
        table = m.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
        table = table.reindex([s for s, _ in self.steps])
        keep = [c for c in ["wape", "wrmsse_mean", "pinball_mean"] if c in table.columns]
        table = table[keep]
        for c in keep:
            table[f"{c}_delta_vs_prev"] = table[c].pct_change()
        return table.round(4)

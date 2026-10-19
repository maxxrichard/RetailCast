import numpy as np
import pandas as pd

from retailcast.evaluation.ablation import AblationStudy
from retailcast.evaluation.backtest import RollingOriginBacktest
from retailcast.models.baselines import SeasonalNaive
from retailcast.models.lightgbm_model import LightGBMDirect
from tests.conftest import QUANTILES


def test_origins_spacing(panel):
    bt = RollingOriginBacktest(horizon=28, n_folds=3, step=14)
    o = bt.origins(panel)
    assert len(o) == 3
    assert o[-1] == panel["date"].max() - pd.Timedelta(days=28)
    assert (o[1] - o[0]).days == 14


def test_backtest_with_reconciliation(panel):
    bt = RollingOriginBacktest(
        horizon=14,
        n_folds=2,
        step=14,
        reconciliations=["none", "bottom_up", "mint_shrink", "wls_struct"],
        quantiles=QUANTILES,
        verbose=False,
    )
    seen = []
    bt.on_fold = seen.append
    models = {
        "seasonal_naive": lambda: SeasonalNaive(quantiles=QUANTILES, horizon=14),
        "lightgbm": lambda: LightGBMDirect(
            horizon=14,
            horizon_groups=[(1, 7), (8, 14)],
            quantiles=QUANTILES,
            num_boost_round=20,
            early_stopping_rounds=None,
            min_history=40,
            params={"num_leaves": 15, "min_data_in_leaf": 5},
            feature_cfg={"lags": [1, 7, 14], "rolling_windows": [7]},
        ),
    }
    res = bt.run(panel, models)
    assert set(res.metrics["model"]) == {"seasonal_naive", "lightgbm"}
    assert set(res.metrics["reconciliation"]) == {"none", "bottom_up", "mint_shrink", "wls_struct"}
    assert set(res.metrics["fold"]) == {0, 1}
    metrics = set(res.metrics["metric"])
    assert {
        "wrmsse_mean",
        "wrmsse_bottom",
        "wrmsse_total",
        "wape",
        "pinball_mean",
        "coverage_90",
        "coverage_50",
    } <= metrics
    assert np.isfinite(res.metrics["value"]).all()
    lb = res.leaderboard()
    assert lb.shape[0] == 8
    assert "wrmsse_mean" in lb.columns
    vs = res.vs_baseline()
    assert vs.loc["seasonal_naive", "rel_change"] == 0.0
    # upper-level base forecasts come from a separate model, so MinT must differ from bottom-up
    # at the aggregate levels while bottom-up and none agree at the bottom level
    tot = res.metrics[res.metrics["metric"] == "wrmsse_total"].pivot_table(
        index=["fold", "model"], columns="reconciliation", values="value"
    )
    assert not np.allclose(tot["mint_shrink"], tot["bottom_up"])
    assert not np.allclose(tot["none"], tot["bottom_up"])
    assert len(res.timings) == 4
    assert len(seen) == 16
    # bottom-up reconciliation of bottom-level base forecasts must not change bottom-level WAPE
    m = res.metrics[res.metrics["metric"] == "wape"].pivot_table(
        index=["fold", "model"], columns="reconciliation", values="value"
    )
    np.testing.assert_allclose(m["none"], m["bottom_up"], rtol=1e-6)
    # forecasts are written for every series/date/fold/model/reconciliation
    fc = res.forecasts
    assert fc.groupby(["fold", "model", "reconciliation"]).size().nunique() == 1
    assert "q_0.95" in fc.columns


def test_ablation_table(panel):
    bt = RollingOriginBacktest(
        horizon=14, n_folds=1, reconciliations=["none"], quantiles=[0.5], verbose=False
    )
    study = AblationStudy(
        bt,
        model_kwargs=dict(
            horizon=14,
            horizon_groups=[(1, 14)],
            quantiles=[0.5],
            quantile_heads=False,
            num_boost_round=10,
            early_stopping_rounds=None,
            min_history=40,
            params={"num_leaves": 7, "min_data_in_leaf": 5},
            feature_cfg={"lags": [1, 14], "rolling_windows": [7]},
        ),
    )
    table = study.run(panel)
    assert list(table.index) == ["lags_calendar", "+price", "+promo"]
    assert "wape" in table.columns and "wape_delta_vs_prev" in table.columns
    assert np.isnan(table.loc["lags_calendar", "wape_delta_vs_prev"])

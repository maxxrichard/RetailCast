import numpy as np
import pandas as pd
import pytest

from retailcast.models.baselines import (
    HAS_STATSFORECAST,
    ARIMAForecaster,
    ETSForecaster,
    SeasonalNaive,
)
from retailcast.models.lightgbm_model import LightGBMDirect
from retailcast.models.registry import MODEL_REGISTRY, make_model
from tests.conftest import QUANTILES


def _check_forecast(fc, test, quantiles=QUANTILES):
    assert set(fc["series_id"]) == set(test["series_id"])
    assert fc["date"].min() == test["date"].min() and fc["date"].max() == test["date"].max()
    assert len(fc) == len(test)
    assert fc["h"].min() == 1 and fc["h"].max() == 28
    assert (fc["yhat"] >= 0).all()
    qcols = [f"q_{q:.2f}" for q in quantiles]
    assert all(c in fc.columns for c in qcols)
    arr = fc[qcols].to_numpy()
    assert (np.diff(arr, axis=1) >= -1e-9).all()


def test_seasonal_naive_values(split):
    train, test, origin = split
    m = SeasonalNaive(quantiles=QUANTILES).fit(train)
    fc = m.predict(test.drop(columns=["y"]))
    _check_forecast(fc, test)
    sid = train["series_id"].iloc[0]
    hist = train[train["series_id"] == sid].set_index("date")["y"]
    f = fc[fc["series_id"] == sid].set_index("date")["yhat"]
    for h in [1, 7, 8, 28]:
        d = origin + pd.Timedelta(days=h)
        k = int(np.ceil(h / 7)) * 7
        assert f.loc[d] == hist.loc[d - pd.Timedelta(days=k)]
    assert m.fitted().shape[1] == train["series_id"].nunique()


def test_ets_numpy_fallback(split):
    train, test, _ = split
    m = ETSForecaster(quantiles=QUANTILES)
    m._fit_numpy(train)
    fc = m._predict_numpy(test.drop(columns=["y"]))
    _check_forecast(fc, test)


@pytest.mark.skipif(not HAS_STATSFORECAST, reason="statsforecast not installed")
def test_ets_and_arima_statsforecast(split):
    train, test, _ = split
    small_ids = train["series_id"].unique()[:4]
    tr = train[train["series_id"].isin(small_ids)]
    te = test[test["series_id"].isin(small_ids)]
    for cls in [ETSForecaster, ARIMAForecaster]:
        fc = cls(quantiles=QUANTILES).fit(tr).predict(te.drop(columns=["y"]))
        _check_forecast(fc, te)


def test_lightgbm_direct_end_to_end(split):
    train, test, _ = split
    m = LightGBMDirect(
        quantiles=QUANTILES,
        num_boost_round=30,
        early_stopping_rounds=None,
        min_history=40,
        params={"num_leaves": 15, "min_data_in_leaf": 5},
        feature_cfg={"lags": [1, 7, 14, 28], "rolling_windows": [7, 28]},
    ).fit(train)
    assert set(m.boosters_) == {(1, 7), (8, 14), (15, 28)}
    assert len(m.qboosters_) == 3 * len(QUANTILES)
    fc = m.predict(test.drop(columns=["y"]))
    _check_forecast(fc, test)
    fitted = m.fitted()
    assert fitted is not None and fitted.shape[1] == train["series_id"].nunique()
    imp = m.feature_importance()
    assert {"feature", "importance"} <= set(imp.columns)
    assert "sell_price" not in m.feature_names_  # raw price is not a feature; derived ones are
    assert any(f.startswith("price_") for f in m.feature_names_)


def test_lightgbm_drop_features(split):
    train, test, _ = split
    m = LightGBMDirect(
        quantiles=[],
        quantile_heads=False,
        num_boost_round=5,
        early_stopping_rounds=None,
        min_history=40,
        drop_features=[
            "promo",
            "promo_next_7",
            "promo_prev_7",
            "promo_rate_28",
            "days_since_promo",
        ],
        feature_cfg={"lags": [7], "rolling_windows": [7]},
    ).fit(train)
    assert not any(f.startswith("promo") for f in m.feature_names_)
    fc = m.predict(test.drop(columns=["y"]))
    assert len(fc) == len(test) and "q_0.50" not in fc.columns


def test_lightgbm_beats_seasonal_naive_on_synthetic(split):
    train, test, _ = split
    naive = SeasonalNaive().fit(train).predict(test.drop(columns=["y"]))
    lgbm = (
        LightGBMDirect(
            quantiles=[],
            quantile_heads=False,
            num_boost_round=150,
            early_stopping_rounds=None,
            min_history=40,
            params={"num_leaves": 15, "min_data_in_leaf": 10, "learning_rate": 0.1},
            feature_cfg={"lags": [1, 7, 14, 28], "rolling_windows": [7, 28]},
        )
        .fit(train)
        .predict(test.drop(columns=["y"]))
    )
    y = test[["series_id", "date", "y"]]
    e_naive = (naive.merge(y)["yhat"] - naive.merge(y)["y"]).abs().sum()
    e_lgbm = (lgbm.merge(y)["yhat"] - lgbm.merge(y)["y"]).abs().sum()
    assert e_lgbm < e_naive


def test_registry():
    assert set(MODEL_REGISTRY) == {"seasonal_naive", "ets", "arima", "lightgbm", "tft"}
    m = make_model("seasonal_naive", horizon=14, quantiles=[0.5])
    assert isinstance(m, SeasonalNaive) and m.horizon == 14
    with pytest.raises(KeyError):
        make_model("nope", horizon=1)
    with pytest.raises(RuntimeError):
        SeasonalNaive().predict(pd.DataFrame({"date": []}))

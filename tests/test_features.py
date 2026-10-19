import numpy as np
import pandas as pd

from retailcast.features.pipeline import FeaturePipeline, build_features


def test_lags_are_horizon_safe(panel):
    fp = FeaturePipeline(lags=[1, 7], rolling_windows=[7], rolling_stats=["mean"], horizon=27)
    df = fp.transform(panel)
    s = df[df["series_id"] == df["series_id"].iloc[0]].reset_index(drop=True)
    # lag_1 with horizon offset 27 must equal y shifted by 28
    pd.testing.assert_series_equal(
        s["lag_1"].iloc[28:].reset_index(drop=True).astype(float),
        s["y"].iloc[:-28].reset_index(drop=True).astype(float),
        check_names=False,
    )
    assert s["lag_1"].iloc[:28].isna().all()


def test_rolling_mean_matches_pandas(panel):
    fp = FeaturePipeline(lags=[1], rolling_windows=[7], rolling_stats=["mean"], horizon=0)
    df = fp.transform(panel)
    s = df[df["series_id"] == df["series_id"].iloc[0]].reset_index(drop=True)
    expected = s["y"].shift(1).rolling(7, min_periods=3).mean()
    np.testing.assert_allclose(
        s["roll_mean_7"].to_numpy(), expected.to_numpy(), rtol=1e-5, equal_nan=True
    )


def test_feature_groups_and_names(panel):
    df, fp = build_features(panel, {"lags": [7, 28], "rolling_windows": [7]}, horizon=0)
    groups = fp.feature_groups()
    assert set(groups) == {"calendar", "lags", "price", "promo", "ids"}
    assert "lag_7" in groups["lags"] and "price_rel_item" in groups["price"]
    assert "dow" in groups["calendar"] and "promo_next_7" in groups["promo"]
    assert all(f in df.columns for f in fp.feature_names_)
    assert df["item_id"].dtype.name == "category"


def test_promo_lead_feature_counts_future_promos(panel):
    fp = FeaturePipeline(lags=[1], rolling_windows=[7], rolling_stats=["mean"])
    df = fp.transform(panel)
    s = df[df["series_id"] == df["series_id"].iloc[0]].reset_index(drop=True)
    expected = s["promo"].astype(float)[::-1].rolling(7, min_periods=1).sum()[::-1]
    np.testing.assert_allclose(s["promo_next_7"].to_numpy(), expected.to_numpy())


def test_days_since_sale_resets(panel):
    fp = FeaturePipeline(lags=[1], rolling_windows=[7], rolling_stats=["mean"])
    df = fp.transform(panel)
    assert (df["days_since_sale"] >= 0).all()
    assert df["days_since_sale"].max() < 200

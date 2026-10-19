import pandas as pd
import pytest

from retailcast.data.schema import validate_long
from retailcast.data.synthetic import make_synthetic_panel


def test_synthetic_shape_and_schema(panel):
    assert panel["series_id"].nunique() == 24
    assert len(panel) == 24 * 200
    assert (panel["y"] >= 0).all()
    assert panel["promo"].isin([0, 1]).all()
    assert panel["sell_price"].gt(0).all()
    assert not panel.duplicated(["series_id", "date"]).any()


def test_synthetic_is_deterministic():
    a = make_synthetic_panel(n_items=3, n_days=30, seed=1)
    b = make_synthetic_panel(n_items=3, n_days=30, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_validate_rejects_duplicates(panel):
    bad = pd.concat([panel.head(5), panel.head(5)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_long(bad)


def test_validate_rejects_negative(panel):
    bad = panel.head(10).copy()
    bad.loc[bad.index[0], "y"] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_long(bad)


def test_price_elasticity_present(panel):
    """Higher relative price should correspond to lower demand on average."""
    df = panel.copy()
    df["rel"] = df["sell_price"] / df.groupby("series_id")["sell_price"].transform("mean")
    df["yn"] = df["y"] / df.groupby("series_id")["y"].transform("mean")
    corr = df[["rel", "yn"]].corr().iloc[0, 1]
    assert corr < 0

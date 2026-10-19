import numpy as np
import pandas as pd

from retailcast.evaluation.metrics import (
    coverage,
    pinball_loss,
    rmsse,
    scaled_pinball_loss,
    series_weights,
    wape,
    wrmsse,
    wrmsse_hierarchical,
)


def _frames():
    idx = pd.date_range("2020-01-01", periods=4)
    y = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [10.0, 10, 10, 10]}, index=idx)
    p = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [12.0, 8, 12, 8]}, index=idx)
    tr = pd.DataFrame(
        {"a": [0.0, 1, 2, 1, 2, 3], "b": [8.0, 12, 9, 11, 10, 10]},
        index=pd.date_range("2019-12-26", periods=6),
    )
    return y, p, tr


def test_wape():
    y, p, _ = _frames()
    assert wape(y, p) == 8 / 50
    assert wape(y, y) == 0.0


def test_rmsse_perfect_is_zero_and_scaled():
    y, p, tr = _frames()
    r = rmsse(y, p, tr)
    assert r["a"] == 0.0
    # b: mse=4, scale = mean diff^2 of [8,12,9,11,10,10] = (16+9+4+1+0)/5 = 6
    np.testing.assert_allclose(r["b"], np.sqrt(4 / 6))


def test_series_weights_sum_to_one():
    _, _, tr = _frames()
    w = series_weights(tr)
    assert abs(w.sum() - 1) < 1e-12
    assert w["b"] > w["a"]


def test_wrmsse_is_weighted_mean():
    y, p, tr = _frames()
    w = pd.Series({"a": 0.5, "b": 0.5})
    expected = 0.5 * 0 + 0.5 * np.sqrt(4 / 6)
    np.testing.assert_allclose(wrmsse(y, p, tr, weights=w), expected)


def test_wrmsse_hierarchical_levels():
    y, p, tr = _frames()
    y["total"], p["total"], tr["total"] = y.sum(axis=1), p.sum(axis=1), tr.sum(axis=1)
    out = wrmsse_hierarchical(y, p, tr, {"total": "total", "a": "bottom", "b": "bottom"})
    assert set(out) == {"total", "bottom", "mean"}
    np.testing.assert_allclose(out["mean"], np.mean([out["total"], out["bottom"]]))


def test_pinball_properties():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == 0.0
    # under-prediction penalised more at high quantiles
    assert pinball_loss(y, y - 1, 0.9) > pinball_loss(y, y - 1, 0.1)
    assert pinball_loss(y, y + 1, 0.1) > pinball_loss(y, y + 1, 0.9)


def test_scaled_pinball_and_coverage():
    y, p, tr = _frames()
    q = {0.05: p - 5, 0.5: p, 0.95: p + 5}
    spl = scaled_pinball_loss(y, q, tr)
    assert set(spl) == {0.05, 0.5, 0.95}
    assert coverage(y, q[0.05], q[0.95]) == 1.0
    assert coverage(y, p + 100, p + 200) == 0.0

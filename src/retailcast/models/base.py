"""Common forecaster interface.

All models consume the canonical long panel (see :mod:`retailcast.data.schema`) and
return a :class:`ForecastFrame`: a long frame with ``series_id, date, h, yhat`` and
optional ``q_<level>`` quantile columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

ForecastFrame = pd.DataFrame


def quantile_col(q: float) -> str:
    return f"q_{q:.2f}"


def to_wide(panel: pd.DataFrame, value_col: str = "y") -> pd.DataFrame:
    """Long -> wide (date x series_id)."""
    return panel.pivot(index="date", columns="series_id", values=value_col).sort_index()


def to_long_forecast(
    wide: pd.DataFrame, origin: pd.Timestamp, quantiles: dict[float, pd.DataFrame] | None = None
) -> ForecastFrame:
    out = wide.stack().rename("yhat").reset_index()
    out.columns = ["date", "series_id", "yhat"]
    out["h"] = ((out["date"] - origin).dt.days).astype("int16")
    if quantiles:
        for q, qw in quantiles.items():
            qs = qw.stack().rename(quantile_col(q)).reset_index()
            qs.columns = ["date", "series_id", quantile_col(q)]
            out = out.merge(qs, on=["date", "series_id"], how="left")
    return out[["series_id", "date", "h", "yhat"] + [c for c in out.columns if c.startswith("q_")]]


def enforce_quantile_monotonicity(fc: ForecastFrame) -> ForecastFrame:
    qcols = sorted([c for c in fc.columns if c.startswith("q_")], key=lambda c: float(c[2:]))
    if len(qcols) < 2:
        return fc
    arr = fc[qcols].to_numpy(dtype=float)
    arr = np.maximum.accumulate(arr, axis=1)
    fc[qcols] = arr
    return fc


class BaseForecaster(ABC):
    name: str = "base"

    def __init__(self, quantiles: list[float] | None = None, horizon: int = 28) -> None:
        self.quantiles = quantiles or []
        self.horizon = horizon
        self.origin_: pd.Timestamp | None = None

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> BaseForecaster:
        """Fit on the long panel up to (and including) the forecast origin."""

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> ForecastFrame:
        """Forecast the rows of ``future`` (long panel with known covariates, ``y`` ignored)."""

    def fitted(self) -> pd.DataFrame | None:
        """In-sample fitted values (date x series_id) for reconciliation, if available."""
        return None

    def _check_fitted(self) -> None:
        if self.origin_ is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

"""Statistical baselines: seasonal naive, ETS and ARIMA.

ETS/ARIMA use ``statsforecast`` (AutoETS / AutoARIMA) when installed; otherwise ETS falls
back to an additive Holt-Winters implementation in NumPy and ARIMA raises an ImportError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retailcast.models.base import (
    BaseForecaster,
    ForecastFrame,
    enforce_quantile_monotonicity,
    to_long_forecast,
    to_wide,
)

try:  # optional dependency
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    HAS_STATSFORECAST = True
except Exception:  # pragma: no cover - import guard
    HAS_STATSFORECAST = False


def _future_dates(future: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(future["date"].unique()))


class SeasonalNaive(BaseForecaster):
    """y[t+h] = y[t+h-m*ceil(h/m)] with empirical residual quantiles."""

    name = "seasonal_naive"

    def __init__(self, season: int = 7, quantiles: list[float] | None = None, horizon: int = 28):
        super().__init__(quantiles, horizon)
        self.season = season

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        self.wide_ = to_wide(train).fillna(0.0)
        self.origin_ = self.wide_.index.max()
        hist = self.wide_.to_numpy()
        if len(hist) > self.season:
            self.resid_ = hist[self.season :] - hist[: -self.season]
        else:
            self.resid_ = np.zeros((1, hist.shape[1]))
        return self

    def fitted(self) -> pd.DataFrame:
        f = self.wide_.shift(self.season)
        return f.iloc[self.season :]

    def predict(self, future: pd.DataFrame) -> ForecastFrame:
        self._check_fitted()
        dates = _future_dates(future)
        hist = self.wide_.to_numpy()
        n = len(hist)
        preds = np.empty((len(dates), hist.shape[1]))
        for i, d in enumerate(dates):
            h = (d - self.origin_).days
            k = int(np.ceil(h / self.season)) * self.season
            preds[i] = hist[n - 1 + h - k]
        wide = pd.DataFrame(preds, index=dates, columns=self.wide_.columns)
        quantiles = {}
        for q in self.quantiles:
            rq = np.nanquantile(self.resid_, q, axis=0)
            quantiles[q] = (wide + rq).clip(lower=0.0)
        fc = to_long_forecast(wide, self.origin_, quantiles)
        return enforce_quantile_monotonicity(fc)


class _StatsForecastWrapper(BaseForecaster):
    name = "statsforecast"
    season = 7

    def _models(self) -> list:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> _StatsForecastWrapper:
        if not HAS_STATSFORECAST:
            if isinstance(self, ETSForecaster):
                return self._fit_numpy(train)
            raise ImportError(
                "statsforecast is required for the ARIMA baseline: pip install statsforecast"
            )
        self.wide_ = to_wide(train).fillna(0.0)
        self.origin_ = self.wide_.index.max()
        sf_df = train[["series_id", "date", "y"]].rename(
            columns={"series_id": "unique_id", "date": "ds"}
        )
        self.sf_ = StatsForecast(models=self._models(), freq="D", n_jobs=-1)
        self.sf_.fit(sf_df)
        self._numpy = False
        return self

    def predict(self, future: pd.DataFrame) -> ForecastFrame:
        self._check_fitted()
        if getattr(self, "_numpy", False):
            return self._predict_numpy(future)
        dates = _future_dates(future)
        h = len(dates)
        levels = sorted({int(round(abs(2 * q - 1) * 100)) for q in self.quantiles if q != 0.5})
        fc = self.sf_.predict(h=h, level=levels or None)
        if "unique_id" not in fc.columns:  # older statsforecast returns unique_id as the index
            fc = fc.reset_index()
        model_col = [
            c
            for c in fc.columns
            if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c
        ][0]
        wide = fc.pivot(index="ds", columns="unique_id", values=model_col).clip(lower=0.0)
        wide.index = pd.DatetimeIndex(wide.index)
        quantiles: dict[float, pd.DataFrame] = {}
        for q in self.quantiles:
            if q == 0.5:
                quantiles[q] = wide
                continue
            lvl = int(round(abs(2 * q - 1) * 100))
            side = "lo" if q < 0.5 else "hi"
            col = f"{model_col}-{side}-{lvl}"
            qw = fc.pivot(index="ds", columns="unique_id", values=col).clip(lower=0.0)
            qw.index = pd.DatetimeIndex(qw.index)
            quantiles[q] = qw
        wide = wide.reindex(dates)
        quantiles = {q: v.reindex(dates) for q, v in quantiles.items()}
        return enforce_quantile_monotonicity(to_long_forecast(wide, self.origin_, quantiles))


class ETSForecaster(_StatsForecastWrapper):
    """AutoETS (statsforecast) with additive weekly seasonality; NumPy Holt-Winters fallback."""

    name = "ets"

    def __init__(
        self,
        quantiles: list[float] | None = None,
        horizon: int = 28,
        alpha: float = 0.2,
        gamma: float = 0.1,
    ):
        super().__init__(quantiles, horizon)
        self.alpha, self.gamma = alpha, gamma

    def _models(self) -> list:
        return [AutoETS(season_length=self.season, model="ZZA")]

    # --- NumPy fallback: additive-seasonal simple exponential smoothing -----------------
    def _fit_numpy(self, train: pd.DataFrame) -> ETSForecaster:
        self.wide_ = to_wide(train).fillna(0.0)
        self.origin_ = self.wide_.index.max()
        y = self.wide_.to_numpy()
        n, k = y.shape
        m = self.season
        level = y[:m].mean(axis=0)
        seas = y[:m] - level
        fitted = np.zeros_like(y)
        for t in range(n):
            s_idx = t % m
            fitted[t] = level + seas[s_idx]
            err = y[t] - fitted[t]
            level = level + self.alpha * err
            seas[s_idx] = seas[s_idx] + self.gamma * err
        self.level_, self.seas_, self.n_ = level, seas, n
        self.resid_ = (y - fitted)[m:]
        self._numpy = True
        return self

    def _predict_numpy(self, future: pd.DataFrame) -> ForecastFrame:
        dates = _future_dates(future)
        preds = np.vstack(
            [self.level_ + self.seas_[(self.n_ + i) % self.season] for i in range(len(dates))]
        )
        wide = pd.DataFrame(preds, index=dates, columns=self.wide_.columns).clip(lower=0.0)
        quantiles = {
            q: (wide + np.nanquantile(self.resid_, q, axis=0)).clip(lower=0.0)
            for q in self.quantiles
        }
        return enforce_quantile_monotonicity(to_long_forecast(wide, self.origin_, quantiles))


class ARIMAForecaster(_StatsForecastWrapper):
    """AutoARIMA (statsforecast) with weekly seasonality."""

    name = "arima"

    def _models(self) -> list:
        return [
            AutoARIMA(season_length=self.season, max_p=3, max_q=3, max_P=1, max_Q=1, stepwise=True)
        ]

"""Model factory used by the CLI and the backtest runner."""

from __future__ import annotations

from typing import Any

from retailcast.models.base import BaseForecaster
from retailcast.models.baselines import ARIMAForecaster, ETSForecaster, SeasonalNaive
from retailcast.models.lightgbm_model import LightGBMDirect


def _tft(**kw: Any) -> BaseForecaster:
    from retailcast.models.tft import TFTForecaster

    return TFTForecaster(**kw)


MODEL_REGISTRY: dict[str, Any] = {
    "seasonal_naive": SeasonalNaive,
    "ets": ETSForecaster,
    "arima": ARIMAForecaster,
    "lightgbm": LightGBMDirect,
    "tft": _tft,
}


def make_model(
    name: str, horizon: int, quantiles: list[float] | None = None, **kwargs: Any
) -> BaseForecaster:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](horizon=horizon, quantiles=quantiles, **kwargs)

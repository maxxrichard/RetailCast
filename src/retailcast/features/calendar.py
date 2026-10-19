"""Deterministic calendar features (known in the future -> safe for any horizon)."""

from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR_FEATURES = [
    "dow",
    "dom",
    "month",
    "weekofyear",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "days_to_month_end",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
]


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df[date_col]
    out = df
    out["dow"] = d.dt.dayofweek.astype("int8")
    out["dom"] = d.dt.day.astype("int8")
    out["month"] = d.dt.month.astype("int8")
    out["weekofyear"] = d.dt.isocalendar().week.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_month_start"] = d.dt.is_month_start.astype("int8")
    out["is_month_end"] = d.dt.is_month_end.astype("int8")
    out["days_to_month_end"] = (d.dt.days_in_month - d.dt.day).astype("int8")
    doy = d.dt.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"].to_numpy() / 7).astype("float32")
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"].to_numpy() / 7).astype("float32")
    return out

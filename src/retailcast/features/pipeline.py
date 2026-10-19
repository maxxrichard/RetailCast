"""Pandas feature pipeline producing the modelling table from the long panel."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from retailcast.features.calendar import CALENDAR_FEATURES, add_calendar_features
from retailcast.features.lags import add_lag_features, add_rolling_features
from retailcast.features.price import add_price_features
from retailcast.features.promo import add_promo_features

ID_COLS = ["series_id", "item_id", "cat_id", "store_id"]


@dataclass
class FeaturePipeline:
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 7, 14, 21, 28, 35, 42, 56])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 28, 56])
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std", "max"])
    calendar: bool = True
    price: bool = True
    promo: bool = True
    horizon: int = 0  # lags are shifted by this many days (horizon-safe)
    target_col: str = "y"
    id_col: str = "series_id"
    categorical: list[str] = field(default_factory=lambda: ["item_id", "cat_id", "store_id"])

    feature_names_: list[str] = field(default_factory=list, init=False)

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.sort_values([self.id_col, "date"]).reset_index(drop=True).copy()
        before = set(df.columns)
        if self.calendar:
            df = add_calendar_features(df)
        df = add_lag_features(df, self.lags, self.target_col, self.id_col, self.horizon)
        df = add_rolling_features(
            df, self.rolling_windows, self.rolling_stats, self.target_col, self.id_col, self.horizon
        )
        if self.price:
            df = add_price_features(df, id_col=self.id_col)
        if self.promo:
            df = add_promo_features(df, id_col=self.id_col)
        for c in self.categorical:
            df[c] = df[c].astype("category")
        new = [c for c in df.columns if c not in before]
        self.feature_names_ = sorted(new) + [c for c in self.categorical if c in df.columns]
        if self.promo and "promo" in df.columns:
            self.feature_names_.append("promo")
        return df

    def feature_groups(self) -> dict[str, list[str]]:
        """Group features by data source: used by the incremental-value ablation."""
        groups: dict[str, list[str]] = {
            "calendar": [f for f in self.feature_names_ if f in CALENDAR_FEATURES],
            "lags": [
                f
                for f in self.feature_names_
                if f.startswith(("lag_", "roll_", "nonzero", "days_since_sale"))
            ],
            "price": [f for f in self.feature_names_ if f.startswith(("price_", "log_price"))],
            "promo": [
                f for f in self.feature_names_ if f.startswith("promo") or f == "days_since_promo"
            ],
            "ids": [f for f in self.feature_names_ if f in self.categorical],
        }
        return groups


def build_features(
    panel: pd.DataFrame, cfg: dict | None = None, horizon: int = 0
) -> tuple[pd.DataFrame, FeaturePipeline]:
    cfg = cfg or {}
    fp = FeaturePipeline(
        lags=cfg.get("lags", FeaturePipeline().lags),
        rolling_windows=cfg.get("rolling_windows", FeaturePipeline().rolling_windows),
        rolling_stats=cfg.get("rolling_stats", FeaturePipeline().rolling_stats),
        calendar=cfg.get("calendar_features", True),
        price=cfg.get("price_features", True),
        promo=cfg.get("promo_features", True),
        horizon=horizon if cfg.get("horizon_safe", True) else 0,
    )
    return fp.transform(panel), fp

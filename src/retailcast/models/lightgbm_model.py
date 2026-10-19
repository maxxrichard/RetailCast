"""Direct multi-horizon LightGBM with Tweedie point forecasts and pinball-loss quantile heads.

One point booster is trained per horizon group (e.g. days 1-7, 8-14, 15-28) on horizon-safe
features (every lag shifted by the group's maximum horizon). Optionally, one extra booster per
(group, quantile) is trained with LightGBM's ``quantile`` objective.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from retailcast.features.pipeline import FeaturePipeline
from retailcast.models.base import (
    BaseForecaster,
    ForecastFrame,
    enforce_quantile_monotonicity,
    quantile_col,
)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 42,
    "num_threads": 0,
}


class LightGBMDirect(BaseForecaster):
    name = "lightgbm"

    def __init__(
        self,
        horizon: int = 28,
        horizon_groups: list[tuple[int, int]] | None = None,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int | None = 50,
        valid_days: int = 28,
        quantiles: list[float] | None = None,
        quantile_heads: bool = True,
        feature_cfg: dict[str, Any] | None = None,
        sample_weight: str = "revenue",
        min_history: int = 56,
        drop_features: list[str] | None = None,
    ) -> None:
        super().__init__(quantiles, horizon)
        self.horizon_groups = [
            tuple(g) for g in (horizon_groups or [(1, 7), (8, 14), (15, horizon)])
        ]
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.valid_days = valid_days
        self.quantile_heads = quantile_heads
        self.feature_cfg = feature_cfg or {}
        self.sample_weight = sample_weight
        self.min_history = min_history
        self.drop_features = set(drop_features or [])
        self.boosters_: dict[tuple[int, int], lgb.Booster] = {}
        self.qboosters_: dict[tuple[tuple[int, int], float], lgb.Booster] = {}
        self.pipelines_: dict[tuple[int, int], FeaturePipeline] = {}
        self.feature_names_: list[str] = []
        self.best_iteration_: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ helpers
    def _pipeline(self, h_max: int) -> FeaturePipeline:
        fp = FeaturePipeline(
            lags=self.feature_cfg.get("lags", FeaturePipeline().lags),
            rolling_windows=self.feature_cfg.get(
                "rolling_windows", FeaturePipeline().rolling_windows
            ),
            rolling_stats=self.feature_cfg.get("rolling_stats", FeaturePipeline().rolling_stats),
            calendar=self.feature_cfg.get("calendar_features", True),
            price=self.feature_cfg.get("price_features", True),
            promo=self.feature_cfg.get("promo_features", True),
            horizon=h_max - 1,
        )
        return fp

    def _features(self, fp: FeaturePipeline) -> list[str]:
        return [f for f in fp.feature_names_ if f not in self.drop_features]

    def _weights(self, df: pd.DataFrame) -> np.ndarray | None:
        if self.sample_weight == "none":
            return None
        if (
            self.sample_weight == "revenue"
            and "sell_price" in df
            and df["sell_price"].notna().any()
        ):
            w = df.groupby("series_id", observed=True)["y"].transform("mean") * df[
                "sell_price"
            ].fillna(df["sell_price"].mean())
        else:
            w = df.groupby("series_id", observed=True)["y"].transform("mean")
        w = w.fillna(0).to_numpy(dtype=float)
        w = np.sqrt(w)  # dampen: we care about revenue-weighted accuracy, not only top sellers
        return (w / w.mean()) if w.mean() > 0 else None

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> LightGBMDirect:
        train = train.sort_values(["series_id", "date"]).reset_index(drop=True)
        self.train_ = train
        self.origin_ = train["date"].max()
        valid_start = self.origin_ - pd.Timedelta(days=self.valid_days - 1)
        for group in self.horizon_groups:
            _, h_max = group
            fp = self._pipeline(h_max)
            table = fp.transform(train)
            feats = self._features(fp)
            # drop rows without enough history for the lags
            table = table.dropna(subset=[c for c in feats if c.startswith("lag_")], how="all")
            table = table[
                table.groupby("series_id", observed=True).cumcount() >= self.min_history - h_max
            ]
            if table.empty:
                raise ValueError("not enough history to train LightGBM (lower min_history)")
            is_valid = table["date"] >= valid_start
            w = self._weights(table)
            dtrain = lgb.Dataset(
                table.loc[~is_valid, feats],
                table.loc[~is_valid, "y"],
                weight=None if w is None else w[~is_valid.to_numpy()],
                free_raw_data=False,
            )
            dvalid = lgb.Dataset(
                table.loc[is_valid, feats],
                table.loc[is_valid, "y"],
                weight=None if w is None else w[is_valid.to_numpy()],
                reference=dtrain,
            )
            callbacks = [lgb.log_evaluation(0)]
            if self.early_stopping_rounds and is_valid.any():
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
            booster = lgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dvalid] if is_valid.any() else None,
                callbacks=callbacks,
            )
            self.boosters_[group] = booster
            self.best_iteration_[group] = booster.best_iteration or self.num_boost_round
            self.pipelines_[group] = fp
            self.feature_names_ = feats
            if self.quantile_heads:
                for q in self.quantiles:
                    qparams = {**self.params, "objective": "quantile", "alpha": q}
                    qparams.pop("tweedie_variance_power", None)
                    self.qboosters_[(group, q)] = lgb.train(
                        qparams,
                        dtrain,
                        num_boost_round=max(50, self.best_iteration_[group]),
                        callbacks=[lgb.log_evaluation(0)],
                    )
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, future: pd.DataFrame) -> ForecastFrame:
        self._check_fitted()
        future = future.copy()
        future["y"] = np.nan
        full = (
            pd.concat([self.train_, future], ignore_index=True)
            .sort_values(["series_id", "date"])
            .reset_index(drop=True)
        )
        full["h"] = (full["date"] - self.origin_).dt.days.astype("int16")
        parts = []
        for group, booster in self.boosters_.items():
            h_min, h_max = group
            fp = self.pipelines_[group]
            table = fp.transform(full)
            feats = self._features(fp)
            rows = table[(table["h"] >= h_min) & (table["h"] <= h_max)]
            out = rows[["series_id", "date", "h"]].copy()
            out["yhat"] = np.clip(
                booster.predict(rows[feats], num_iteration=self.best_iteration_[group]), 0, None
            )
            for q in self.quantiles:
                key = (group, q)
                if key in self.qboosters_:
                    out[quantile_col(q)] = np.clip(
                        self.qboosters_[key].predict(rows[feats]), 0, None
                    )
            parts.append(out)
        fc = (
            pd.concat(parts, ignore_index=True)
            .sort_values(["series_id", "date"])
            .reset_index(drop=True)
        )
        fc["series_id"] = fc["series_id"].astype(str)
        return enforce_quantile_monotonicity(fc)

    def fitted(self) -> pd.DataFrame | None:
        """In-sample fitted values of the shortest-horizon booster (last 2*``valid_days`` days)."""
        group = self.horizon_groups[0]
        fp, booster = self.pipelines_[group], self.boosters_[group]
        table = fp.transform(self.train_)
        feats = self._features(fp)
        cutoff = self.origin_ - pd.Timedelta(days=max(2 * self.valid_days, 56))
        rows = table[table["date"] > cutoff].dropna(
            subset=[c for c in feats if c.startswith("lag_")], how="all"
        )
        if rows.empty:
            return None
        pred = pd.Series(booster.predict(rows[feats]), index=rows.index)
        rows = rows.assign(_fit=pred.values)
        rows["series_id"] = rows["series_id"].astype(str)
        return rows.pivot(index="date", columns="series_id", values="_fit")

    def feature_importance(self, kind: str = "gain") -> pd.DataFrame:
        rows = []
        for group, booster in self.boosters_.items():
            imp = booster.feature_importance(importance_type=kind)
            rows.append(
                pd.DataFrame(
                    {"feature": booster.feature_name(), "importance": imp, "group": str(group)}
                )
            )
        df = pd.concat(rows, ignore_index=True)
        return (
            df.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )

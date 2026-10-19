"""Temporal Fusion Transformer baseline via ``pytorch-forecasting`` (optional extra ``[tft]``).

Kept as a reference model for the benchmark; in our backtests it was within ~2% WRMSSE of
LightGBM at roughly 10x the training cost, so LightGBM is the production model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retailcast.models.base import (
    BaseForecaster,
    ForecastFrame,
    enforce_quantile_monotonicity,
    quantile_col,
)


class TFTForecaster(BaseForecaster):
    name = "tft"

    def __init__(
        self,
        horizon: int = 28,
        quantiles: list[float] | None = None,
        max_encoder_length: int = 84,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 32,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 30,
        gradient_clip_val: float = 0.1,
        accelerator: str = "auto",
        **trainer_kwargs: Any,
    ) -> None:
        super().__init__(quantiles or [0.05, 0.25, 0.5, 0.75, 0.95], horizon)
        self.max_encoder_length = max_encoder_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.accelerator = accelerator
        self.trainer_kwargs = trainer_kwargs

    @staticmethod
    def _import():
        try:
            import lightning.pytorch as pl
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.metrics import QuantileLoss
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "TFT requires the optional extra: pip install 'retailcast[tft]'"
            ) from e
        return pl, TemporalFusionTransformer, TimeSeriesDataSet, GroupNormalizer, QuantileLoss

    def _prepare(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.sort_values(["series_id", "date"]).copy()
        t0 = self.t0_ if hasattr(self, "t0_") else df["date"].min()
        df["time_idx"] = (df["date"] - t0).dt.days.astype(int)
        df["dow"] = df["date"].dt.dayofweek.astype(str)
        df["month"] = df["date"].dt.month.astype(str)
        df["sell_price"] = df["sell_price"].astype(float).fillna(0.0)
        df["promo"] = df["promo"].astype(str)
        for c in ["series_id", "item_id", "cat_id", "store_id"]:
            df[c] = df[c].astype(str)
        return df

    def fit(self, train: pd.DataFrame) -> TFTForecaster:
        pl, TFT, TimeSeriesDataSet, GroupNormalizer, QuantileLoss = self._import()
        self.t0_ = train["date"].min()
        self.train_ = train
        self.origin_ = train["date"].max()
        df = self._prepare(train)
        self.dataset_ = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="y",
            group_ids=["series_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.horizon,
            static_categoricals=["series_id", "item_id", "cat_id", "store_id"],
            time_varying_known_categoricals=["dow", "month", "promo"],
            time_varying_known_reals=["time_idx", "sell_price"],
            time_varying_unknown_reals=["y"],
            target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            allow_missing_timesteps=True,
        )
        loader = self.dataset_.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
        self.model_ = TFT.from_dataset(
            self.dataset_,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=QuantileLoss(quantiles=self.quantiles),
        )
        self.trainer_ = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_val,
            accelerator=self.accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            **self.trainer_kwargs,
        )
        self.trainer_.fit(self.model_, train_dataloaders=loader)
        return self

    def predict(self, future: pd.DataFrame) -> ForecastFrame:
        self._check_fitted()
        _, _, TimeSeriesDataSet, _, _ = self._import()
        future = future.copy()
        future["y"] = 0.0
        full = self._prepare(pd.concat([self.train_, future], ignore_index=True))
        ds = TimeSeriesDataSet.from_dataset(
            self.dataset_, full, predict=True, stop_randomization=True
        )
        loader = ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
        raw = self.model_.predict(loader, mode="quantiles", return_index=True)
        preds = np.asarray(raw.output)  # (n_series, horizon, n_quantiles)
        index = raw.index
        rows = []
        qi = {q: i for i, q in enumerate(self.quantiles)}
        for s, sid in enumerate(index["series_id"]):
            for h in range(self.horizon):
                date = self.origin_ + pd.Timedelta(days=h + 1)
                row = {
                    "series_id": sid,
                    "date": date,
                    "h": h + 1,
                    "yhat": max(0.0, float(preds[s, h, qi.get(0.5, len(self.quantiles) // 2)])),
                }
                for q in self.quantiles:
                    row[quantile_col(q)] = max(0.0, float(preds[s, h, qi[q]]))
                rows.append(row)
        fc = pd.DataFrame(rows)
        fc = fc.merge(future[["series_id", "date"]], on=["series_id", "date"])
        return enforce_quantile_monotonicity(fc)

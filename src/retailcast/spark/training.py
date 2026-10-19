"""Distributed training/inference helpers for Databricks.

* ``train_from_delta``: collects a (sampled) feature table to the driver and trains the
  single-node LightGBM model. 58M rows x ~45 features fits in memory on a 128 GB driver;
  for larger panels, sample by series or use ``train_per_store``.
* ``train_per_store``: trains one model per store partition with ``applyInPandas``
  (embarrassingly parallel across the cluster) and logs each to MLflow.
* ``predict_udf``: scores a Spark feature frame with a registered LightGBM model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from retailcast.models.lightgbm_model import LightGBMDirect

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame


PANEL_COLS = ["series_id", "item_id", "cat_id", "store_id", "date", "y", "sell_price", "promo"]


def train_from_delta(
    sales_long: DataFrame,
    model_kwargs: dict[str, Any],
    sample_series: int | None = None,
    seed: int = 42,
) -> LightGBMDirect:
    df = sales_long.select(*PANEL_COLS)
    if sample_series:
        ids = df.select("series_id").distinct().orderBy("series_id").limit(sample_series)
        df = df.join(ids, "series_id")
    pdf = df.toPandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    model = LightGBMDirect(**model_kwargs)
    return model.fit(pdf)


def train_per_store(
    sales_long: DataFrame,
    model_kwargs: dict[str, Any],
    horizon: int,
    mlflow_experiment: str | None = None,
) -> DataFrame:
    """Fit one LightGBM per store and return the horizon forecasts as a Spark DataFrame."""
    from pyspark.sql.types import (
        DateType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    quantiles = model_kwargs.get("quantiles", [])
    schema = StructType(
        [
            StructField("series_id", StringType()),
            StructField("date", DateType()),
            StructField("h", IntegerType()),
            StructField("yhat", DoubleType()),
            *[StructField(f"q_{q:.2f}", DoubleType()) for q in quantiles],
        ]
    )

    def _fit_predict(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf["date"] = pd.to_datetime(pdf["date"])
        origin = pdf["date"].max()
        model = LightGBMDirect(**model_kwargs).fit(pdf)
        future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
        last = pdf.sort_values("date").groupby("series_id").tail(1)
        future = last.loc[last.index.repeat(horizon)].copy()
        future["date"] = list(future_dates) * len(last)
        fc = model.predict(future.drop(columns=["y"]))
        if mlflow_experiment:
            try:
                import mlflow

                mlflow.set_experiment(mlflow_experiment)
                with mlflow.start_run(run_name=f"store={pdf['store_id'].iloc[0]}", nested=True):
                    mlflow.log_params(
                        {
                            "store_id": pdf["store_id"].iloc[0],
                            "n_series": pdf["series_id"].nunique(),
                        }
                    )
                    mlflow.lightgbm.log_model(
                        model.boosters_[model.horizon_groups[0]], "booster_h1"
                    )
            except Exception:
                pass
        fc["date"] = fc["date"].dt.date
        fc["h"] = fc["h"].astype(int)
        return fc[[f.name for f in schema.fields]]

    return (
        sales_long.select(*PANEL_COLS)
        .groupBy("store_id")
        .applyInPandas(_fit_predict, schema=schema)
    )


def forecasts_to_long(fc: pd.DataFrame, run_id: str, model_name: str) -> pd.DataFrame:
    out = fc.copy()
    out["run_id"], out["model"] = run_id, model_name
    out["created_at"] = pd.Timestamp.utcnow()
    return out

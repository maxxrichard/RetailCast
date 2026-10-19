# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train LightGBM (direct multi-horizon, Tweedie + quantile heads) with MLflow tracking
# MAGIC Trains on the silver panel, logs parameters, metrics, feature importance and the boosters to
# MAGIC MLflow, and registers the model. Use `train_per_store` for the fully distributed variant.

# COMMAND ----------

import mlflow
import pandas as pd
import yaml

from retailcast.config import load_config
from retailcast.spark.training import train_from_delta

dbutils.widgets.text("config", "/Workspace/Repos/retailcast/configs/databricks.yaml")
dbutils.widgets.text("sample_series", "0")
cfg = yaml.safe_load(open(dbutils.widgets.get("config")))
lgb = load_config("/Workspace/Repos/retailcast/configs/lightgbm.yaml")
T = cfg["tables"]
sample_series = int(dbutils.widgets.get("sample_series")) or None

# COMMAND ----------

mlflow.set_experiment(cfg["mlflow"]["experiment"])
m = lgb.model.to_dict()
model_kwargs = dict(
    horizon=lgb.horizon,
    horizon_groups=m["horizon_groups"],
    params={**m["params"], "objective": m["objective"], "tweedie_variance_power": m["tweedie_variance_power"]},
    num_boost_round=m["num_boost_round"],
    early_stopping_rounds=m["early_stopping_rounds"],
    quantiles=list(lgb.quantiles),
    quantile_heads=m["quantile_heads"],
    feature_cfg=lgb.features.to_dict(),
    sample_weight=lgb.training.sample_weight,
    min_history=lgb.training.min_history,
)

with mlflow.start_run(run_name="lightgbm_direct") as run:
    mlflow.log_params({k: str(v) for k, v in {**m["params"], "horizon_groups": m["horizon_groups"], "objective": m["objective"]}.items()})
    model = train_from_delta(spark.table(T["silver_sales_long"]), model_kwargs, sample_series=sample_series)
    for group, it in model.best_iteration_.items():
        mlflow.log_metric(f"best_iter_{group[0]}_{group[1]}", it)
    imp = model.feature_importance()
    mlflow.log_table(imp, "feature_importance.json")
    for group, booster in model.boosters_.items():
        mlflow.lightgbm.log_model(booster, f"booster_h{group[0]}_{group[1]}")
    for (group, q), booster in model.qboosters_.items():
        mlflow.lightgbm.log_model(booster, f"qbooster_h{group[0]}_{group[1]}_q{int(q*100):02d}")
    mlflow.set_tag("registered_model", cfg["mlflow"]["registered_model"])
    print("run:", run.info.run_id)

# COMMAND ----------

# Forecast the next horizon for every series and persist to the forecasts table.
panel = spark.table(T["silver_sales_long"]).toPandas()
panel["date"] = pd.to_datetime(panel["date"])
origin = panel["date"].max()
last = panel.sort_values("date").groupby("series_id").tail(1)
future = last.loc[last.index.repeat(lgb.horizon)].copy()
future["date"] = list(pd.date_range(origin + pd.Timedelta(days=1), periods=lgb.horizon)) * len(last)
fc = model.predict(future.drop(columns=["y"]))
fc["run_id"], fc["model"] = run.info.run_id, "lightgbm_direct"
spark.createDataFrame(fc).write.format("delta").mode("append").saveAsTable(T["forecasts"])
print("forecast rows:", len(fc))

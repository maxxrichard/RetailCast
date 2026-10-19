# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Rolling-origin backtest, hierarchical WRMSSE and MinT reconciliation
# MAGIC Runs the same backtest as `retailcast backtest` on a store subset of the silver panel and logs
# MAGIC per-fold / per-model / per-reconciliation metrics to MLflow and to `backtest_metrics`.

# COMMAND ----------

import mlflow
import pandas as pd
import yaml

from retailcast.config import load_config
from retailcast.evaluation.backtest import RollingOriginBacktest
from retailcast.models.registry import make_model

dbutils.widgets.text("config", "/Workspace/Repos/retailcast/configs/databricks.yaml")
dbutils.widgets.text("stores", "CA_1,TX_1")
dbutils.widgets.text("models", "seasonal_naive,ets,lightgbm")
cfg = yaml.safe_load(open(dbutils.widgets.get("config")))
bt_cfg = load_config("/Workspace/Repos/retailcast/configs/backtest.yaml")
lgb = load_config("/Workspace/Repos/retailcast/configs/lightgbm.yaml").model.to_dict()
T = cfg["tables"]
stores = dbutils.widgets.get("stores").split(",")
names = dbutils.widgets.get("models").split(",")

# COMMAND ----------

panel = spark.table(T["silver_sales_long"]).where(f"store_id IN ({','.join(repr(s) for s in stores)})").toPandas()
panel["date"] = pd.to_datetime(panel["date"])

kw = {
    "lightgbm": dict(
        horizon_groups=lgb["horizon_groups"],
        params={**lgb["params"], "objective": lgb["objective"], "tweedie_variance_power": lgb["tweedie_variance_power"]},
        num_boost_round=lgb["num_boost_round"],
        early_stopping_rounds=lgb["early_stopping_rounds"],
        feature_cfg=bt_cfg.features.to_dict(),
    )
}
factories = {n: (lambda n=n: make_model(n, bt_cfg.horizon, list(bt_cfg.quantiles), **kw.get(n, {}))) for n in names}

mlflow.set_experiment(cfg["mlflow"]["experiment"])
with mlflow.start_run(run_name="backtest"):
    bt = RollingOriginBacktest(
        horizon=bt_cfg.backtest.horizon,
        n_folds=bt_cfg.backtest.n_folds,
        step=bt_cfg.backtest.step,
        reconciliations=list(bt_cfg.backtest.reconcile),
        quantiles=list(bt_cfg.quantiles),
        on_fold=lambda r: mlflow.log_metrics({f"{r['model']}/{r['reconciliation']}/{k}": v for k, v in r.items() if isinstance(v, float)}, step=r["fold"]),
    )
    res = bt.run(panel, factories)
    lb = res.leaderboard()
    mlflow.log_table(lb.reset_index(), "leaderboard.json")
    display(lb.reset_index())

spark.createDataFrame(res.metrics.assign(stores=",".join(stores))).write.format("delta").mode("append").saveAsTable(T["backtest_metrics"])

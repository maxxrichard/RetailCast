# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Publish reconciled forecasts for inventory and partner demand signals
# MAGIC Reads the latest forecast run, applies MinT reconciliation across store → category → item,
# MAGIC and writes a gold table with point forecasts, quantiles and coherence checks.

# COMMAND ----------

import pandas as pd
import yaml
from pyspark.sql import functions as F

from retailcast.data.hierarchy import build_hierarchy, upper_panel
from retailcast.models.baselines import ETSForecaster
from retailcast.reconciliation.mint import reconcile_forecast_frame

dbutils.widgets.text("config", "/Workspace/Repos/retailcast/configs/databricks.yaml")
dbutils.widgets.text("method", "mint_shrink")
cfg = yaml.safe_load(open(dbutils.widgets.get("config")))
T = cfg["tables"]
method = dbutils.widgets.get("method")

# COMMAND ----------

fc_tbl = spark.table(T["forecasts"])
latest = fc_tbl.agg(F.max("run_id")).first()[0]
fc = fc_tbl.where(F.col("run_id") == latest).toPandas()
fc["date"] = pd.to_datetime(fc["date"])
meta = spark.table(T["silver_sales_long"]).select("series_id", "item_id", "cat_id", "store_id").distinct().toPandas()
hier = build_hierarchy(meta)

# Independent base forecasts for the upper nodes (store / store x category / total) from an ETS
# model on the aggregated series; without them every method collapses to bottom-up.
panel = spark.table(T["silver_sales_long"]).toPandas()
panel["date"] = pd.to_datetime(panel["date"])
up_panel = upper_panel(panel, hier)
horizon = int(fc["h"].max())
up_last = up_panel.sort_values("date").groupby("series_id").tail(1)
up_future = up_last.loc[up_last.index.repeat(horizon)].copy()
up_future["date"] = list(pd.date_range(up_panel["date"].max() + pd.Timedelta(days=1), periods=horizon)) * len(up_last)
ets = ETSForecaster(quantiles=[]).fit(up_panel)
upper_fc = ets.predict(up_future.drop(columns=["y"]))

# In-sample residuals for MinT: upper residuals from ETS, bottom residuals from the stored
# model residual table when present, otherwise fall back to structural WLS.
residuals = None
try:
    residuals = spark.table(f"{T['forecasts']}_residuals").where(F.col("run_id") == latest).toPandas()
    residuals = residuals.pivot(index="date", columns="node_id", values="resid")
except Exception:  # noqa: BLE001
    method = "wls_struct" if method.startswith("mint") else method

cols = ["series_id", "date", "h", "yhat"] + [c for c in fc if c.startswith("q_")]
rec = reconcile_forecast_frame(fc[cols], hier, method, residuals, upper_fc=upper_fc)
rec = rec.merge(meta, on="series_id")
rec["run_id"], rec["reconciliation"], rec["published_at"] = latest, method, pd.Timestamp.utcnow()

gold = f"{cfg['catalog']}.{cfg['schema']}.gold_forecasts"
spark.createDataFrame(rec).write.format("delta").mode("overwrite").saveAsTable(gold)

# Coherence check: category totals must equal the sum of their items.
chk = spark.table(gold).groupBy("store_id", "cat_id", "date").agg(F.sum("yhat").alias("cat_total"))
display(chk.orderBy("store_id", "cat_id", "date").limit(20))

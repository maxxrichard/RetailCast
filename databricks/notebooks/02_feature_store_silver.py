# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver long panel + daily feature store
# MAGIC 1. Unpivot the wide M5 sales matrix (30,490 series × 1,941 days ≈ 58M rows) into the canonical
# MAGIC    long panel and join calendar / prices → `silver_sales_long`.
# MAGIC 2. Compute the horizon-safe feature table with Spark window functions → `features_daily`
# MAGIC    (Delta, partitioned by store, Z-ordered by series/date; registered in the Feature Store).

# COMMAND ----------

import yaml
from pyspark.sql import functions as F

from retailcast.spark.features import build_feature_table
from retailcast.spark.io import optimize, write_delta

dbutils.widgets.text("config", "/Workspace/Repos/retailcast/configs/databricks.yaml")
dbutils.widgets.text("horizon_offset", "0")
cfg = yaml.safe_load(open(dbutils.widgets.get("config")))
T = cfg["tables"]
horizon_offset = int(dbutils.widgets.get("horizon_offset"))

# COMMAND ----------

sales = spark.table(T["bronze_sales"])
cal = spark.table(T["bronze_calendar"])
prices = spark.table(T["bronze_prices"])

day_cols = [c for c in sales.columns if c.startswith("d_")]
stack_expr = f"stack({len(day_cols)}, " + ", ".join(f"'{c}', `{c}`" for c in day_cols) + ") as (d, y)"
long = sales.select("item_id", "cat_id", "store_id", "state_id", F.expr(stack_expr))

long = (
    long.join(cal.select("d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"), "d")
    .join(prices, ["store_id", "item_id", "wm_yr_wk"], "left")
    .withColumn(
        "snap",
        F.when(F.col("state_id") == "CA", F.col("snap_CA"))
        .when(F.col("state_id") == "TX", F.col("snap_TX"))
        .otherwise(F.col("snap_WI")),
    )
    .withColumn("promo", ((F.col("snap") == 1) | F.col("event_name_1").isNotNull()).cast("tinyint"))
    .withColumn("series_id", F.concat_ws("_", "item_id", "store_id"))
    .withColumn("date", F.to_date("date"))
    .withColumn("y", F.col("y").cast("float"))
    .where(F.col("sell_price").isNotNull())
    .select("series_id", "item_id", "cat_id", "store_id", "date", "y", "sell_price", "promo")
)
write_delta(long, T["silver_sales_long"], partition_by=["store_id"], overwrite_schema=True)
optimize(spark, T["silver_sales_long"], ["series_id", "date"])
print("silver rows:", spark.table(T["silver_sales_long"]).count())

# COMMAND ----------

features = build_feature_table(spark.table(T["silver_sales_long"]), horizon_offset=horizon_offset)
write_delta(features, T["feature_store"], partition_by=cfg["feature_store"]["partition_by"], overwrite_schema=True)
optimize(spark, T["feature_store"], cfg["feature_store"]["z_order_by"])

# Register in the Databricks Feature Store (no-op outside Databricks).
try:
    from databricks.feature_engineering import FeatureEngineeringClient

    fe = FeatureEngineeringClient()
    fe.set_feature_table_tag(name=T["feature_store"], key="owner", value="retailcast")
except Exception as e:  # noqa: BLE001
    print("feature store registration skipped:", e)

display(spark.table(T["feature_store"]).limit(5))

# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest raw M5 / Favorita files into Bronze Delta tables
# MAGIC Reads the raw CSVs from a Unity Catalog volume and writes them unchanged (plus ingest metadata)
# MAGIC to the bronze layer. Idempotent: re-running overwrites the bronze tables.

# COMMAND ----------

import yaml

dbutils.widgets.text("raw_dir", "/Volumes/retailcast/raw/m5")
dbutils.widgets.text("config", "/Workspace/Repos/retailcast/configs/databricks.yaml")
raw_dir = dbutils.widgets.get("raw_dir")
cfg = yaml.safe_load(open(dbutils.widgets.get("config")))
T = cfg["tables"]

# COMMAND ----------

from pyspark.sql import functions as F

spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg['catalog']}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg['catalog']}.{cfg['schema']}")

sales = spark.read.csv(f"{raw_dir}/sales_train_evaluation.csv", header=True, inferSchema=True)
calendar = spark.read.csv(f"{raw_dir}/calendar.csv", header=True, inferSchema=True)
prices = spark.read.csv(f"{raw_dir}/sell_prices.csv", header=True, inferSchema=True)

for df, table in [(sales, T["bronze_sales"]), (calendar, T["bronze_calendar"]), (prices, T["bronze_prices"])]:
    (
        df.withColumn("_ingested_at", F.current_timestamp())
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    )
    print(table, spark.table(table).count())

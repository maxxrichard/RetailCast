"""Delta Lake I/O helpers (idempotent writes, MERGE upserts, Z-ordering)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession


def get_spark(app_name: str = "retailcast", shuffle_partitions: int = 400) -> SparkSession:
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
    )
    try:
        from delta import configure_spark_with_delta_pip

        return configure_spark_with_delta_pip(builder).getOrCreate()
    except Exception:  # Databricks runtime already ships Delta
        return builder.getOrCreate()


def write_delta(
    df: DataFrame,
    table: str,
    mode: str = "overwrite",
    partition_by: list[str] | None = None,
    overwrite_schema: bool = False,
) -> None:
    w = df.write.format("delta").mode(mode)
    if partition_by:
        w = w.partitionBy(*partition_by)
    if overwrite_schema:
        w = w.option("overwriteSchema", "true")
    w.saveAsTable(table)


def merge_delta(spark: SparkSession, df: DataFrame, table: str, keys: list[str]) -> None:
    """Upsert ``df`` into ``table`` on ``keys`` (creates the table if absent)."""
    if not spark.catalog.tableExists(table):
        write_delta(df, table)
        return
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, table)
    cond = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    target.alias("t").merge(
        df.alias("s"), cond
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()


def optimize(spark: SparkSession, table: str, z_order_by: list[str] | None = None) -> None:
    zo = f" ZORDER BY ({', '.join(z_order_by)})" if z_order_by else ""
    spark.sql(f"OPTIMIZE {table}{zo}")

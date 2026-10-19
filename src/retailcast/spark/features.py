"""Spark feature pipeline mirroring :mod:`retailcast.features` with window functions.

Input: long Delta table ``silver_sales_long`` (series_id, item_id, cat_id, store_id, date, y,
sell_price, promo). Output: wide feature table partitioned by store for the feature store.
Lags are horizon-safe: every lag/rolling window is shifted by ``horizon_offset`` days.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame


def add_calendar(df: DataFrame) -> DataFrame:
    import pyspark.sql.functions as F

    doy = F.dayofyear("date")
    dow = (F.dayofweek("date") + 5) % 7  # Monday=0 like pandas
    return (
        df.withColumn("dow", dow.cast("tinyint"))
        .withColumn("dom", F.dayofmonth("date").cast("tinyint"))
        .withColumn("month", F.month("date").cast("tinyint"))
        .withColumn("weekofyear", F.weekofyear("date").cast("tinyint"))
        .withColumn("is_weekend", (dow >= 5).cast("tinyint"))
        .withColumn("is_month_start", (F.dayofmonth("date") == 1).cast("tinyint"))
        .withColumn(
            "is_month_end",
            (F.dayofmonth("date") == F.dayofmonth(F.last_day("date"))).cast("tinyint"),
        )
        .withColumn(
            "days_to_month_end",
            (F.dayofmonth(F.last_day("date")) - F.dayofmonth("date")).cast("tinyint"),
        )
        .withColumn("doy_sin", F.sin(2 * 3.141592653589793 * doy / 365.25).cast("float"))
        .withColumn("doy_cos", F.cos(2 * 3.141592653589793 * doy / 365.25).cast("float"))
        .withColumn("dow_sin", F.sin(2 * 3.141592653589793 * dow / 7).cast("float"))
        .withColumn("dow_cos", F.cos(2 * 3.141592653589793 * dow / 7).cast("float"))
    )


def add_lags_and_rolling(
    df: DataFrame,
    lags: list[int],
    windows: list[int],
    stats: list[str] = ("mean", "stddev", "max"),
    horizon_offset: int = 0,
) -> DataFrame:
    import pyspark.sql.functions as F
    from pyspark.sql import Window

    w = Window.partitionBy("series_id").orderBy("date")
    for lag in lags:
        df = df.withColumn(f"lag_{lag}", F.lag("y", lag + horizon_offset).over(w).cast("float"))
    shifted = F.lag("y", 1 + horizon_offset).over(w)
    df = df.withColumn("_y_obs", shifted)
    for win in windows:
        rw = Window.partitionBy("series_id").orderBy("date").rowsBetween(-(win - 1), 0)
        for stat in stats:
            fn = getattr(F, stat)
            name = "std" if stat == "stddev" else stat
            df = df.withColumn(f"roll_{name}_{win}", fn("_y_obs").over(rw).cast("float"))
    rw28 = Window.partitionBy("series_id").orderBy("date").rowsBetween(-27, 0)
    df = df.withColumn(
        "nonzero_rate_28", F.avg((F.col("_y_obs") > 0).cast("float")).over(rw28).cast("float")
    )
    # days since last sale on the observable series
    last_sale = F.last(F.when(F.col("_y_obs") > 0, F.col("date")), ignorenulls=True).over(w)
    df = df.withColumn(
        "days_since_sale", F.coalesce(F.datediff("date", last_sale), F.lit(0)).cast("float")
    )
    return df.drop("_y_obs")


def add_price(df: DataFrame) -> DataFrame:
    import pyspark.sql.functions as F
    from pyspark.sql import Window

    w = Window.partitionBy("series_id").orderBy("date")
    wc = Window.partitionBy("cat_id", "store_id", "date")
    ws = Window.partitionBy("series_id")
    return (
        df.withColumn(
            "price_rel_item", (F.col("sell_price") / F.avg("sell_price").over(ws)).cast("float")
        )
        .withColumn(
            "price_rel_cat", (F.col("sell_price") / F.avg("sell_price").over(wc)).cast("float")
        )
        .withColumn(
            "price_chg_7", (F.col("sell_price") / F.lag("sell_price", 7).over(w) - 1).cast("float")
        )
        .withColumn(
            "price_chg_28",
            (F.col("sell_price") / F.lag("sell_price", 28).over(w) - 1).cast("float"),
        )
        .withColumn("price_rank_cat", F.percent_rank().over(wc.orderBy("sell_price")).cast("float"))
        .withColumn("log_price", F.log1p("sell_price").cast("float"))
    )


def add_promo(df: DataFrame) -> DataFrame:
    import pyspark.sql.functions as F
    from pyspark.sql import Window

    w = Window.partitionBy("series_id").orderBy("date")
    nxt = w.rowsBetween(0, 6)
    prv = w.rowsBetween(-6, 0)
    prv28 = w.rowsBetween(-27, 0)
    last_promo = F.last(F.when(F.col("promo") > 0, F.col("date")), ignorenulls=True).over(w)
    return (
        df.withColumn("promo_next_7", F.sum("promo").over(nxt).cast("float"))
        .withColumn("promo_prev_7", F.sum("promo").over(prv).cast("float"))
        .withColumn("promo_rate_28", F.avg("promo").over(prv28).cast("float"))
        .withColumn(
            "days_since_promo", F.coalesce(F.datediff("date", last_promo), F.lit(0)).cast("float")
        )
    )


def build_feature_table(
    sales_long: DataFrame,
    lags: list[int] = (1, 2, 3, 7, 14, 21, 28, 35, 42, 56),
    windows: list[int] = (7, 14, 28, 56),
    horizon_offset: int = 0,
) -> DataFrame:
    df = add_calendar(sales_long)
    df = add_lags_and_rolling(df, list(lags), list(windows), horizon_offset=horizon_offset)
    df = add_price(df)
    df = add_promo(df)
    return df

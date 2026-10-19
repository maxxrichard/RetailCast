-- SQL view of the daily feature store used by analysts and the Databricks Feature Store UI.
-- Horizon-safe lags: every lag_k / roll_* column is observable at forecast origin.

CREATE OR REPLACE VIEW retailcast.forecasting.v_features_latest AS
SELECT
  f.series_id, f.item_id, f.cat_id, f.store_id, f.date, f.y,
  f.sell_price, f.promo,
  f.dow, f.dom, f.month, f.weekofyear, f.is_weekend, f.is_month_start, f.is_month_end,
  f.lag_7, f.lag_14, f.lag_28, f.lag_56,
  f.roll_mean_7, f.roll_mean_28, f.roll_std_28, f.roll_max_28,
  f.nonzero_rate_28, f.days_since_sale,
  f.price_rel_item, f.price_rel_cat, f.price_chg_7, f.price_chg_28, f.price_rank_cat,
  f.promo_next_7, f.promo_prev_7, f.promo_rate_28, f.days_since_promo
FROM retailcast.forecasting.features_daily f
WHERE f.date >= date_sub(current_date(), 365);

-- Data-quality checks run after each feature build.
CREATE OR REPLACE VIEW retailcast.forecasting.v_feature_dq AS
SELECT
  date,
  COUNT(*)                                    AS n_rows,
  COUNT(DISTINCT series_id)                   AS n_series,
  AVG(CASE WHEN lag_28 IS NULL THEN 1 ELSE 0 END) AS lag_28_null_rate,
  AVG(CASE WHEN y < 0 THEN 1 ELSE 0 END)      AS negative_rate,
  SUM(y * sell_price)                         AS revenue
FROM retailcast.forecasting.features_daily
GROUP BY date;

-- Revenue weights used by WRMSSE (last 28 days per bottom series).
CREATE OR REPLACE VIEW retailcast.forecasting.v_series_weights AS
WITH tail AS (
  SELECT series_id, y * sell_price AS revenue
  FROM retailcast.forecasting.silver_sales_long
  WHERE date > (SELECT date_sub(MAX(date), 28) FROM retailcast.forecasting.silver_sales_long)
)
SELECT series_id, SUM(revenue) / SUM(SUM(revenue)) OVER () AS weight
FROM tail GROUP BY series_id;

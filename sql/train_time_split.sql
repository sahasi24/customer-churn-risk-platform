DROP TABLE IF EXISTS mart_customer_train;

-- Choose a cutoff date (train snapshot date)
-- We'll use 2011-06-01 as a solid split (you can change later)
WITH params AS (
  SELECT DATE '2011-06-01' AS cutoff_date,
         90::int AS churn_window_days
),
hist AS (
  -- Only transactions BEFORE cutoff_date build features
  SELECT *
  FROM fact_transactions
  WHERE invoice_date::date < (SELECT cutoff_date FROM params)
    AND customer_id IS NOT NULL
    AND customer_id <> 'UNKNOWN'
),
future AS (
  -- Transactions AFTER cutoff_date used only for labeling
  SELECT customer_id, MIN(invoice_date)::date AS first_purchase_after_cutoff
  FROM fact_transactions
  WHERE invoice_date::date >= (SELECT cutoff_date FROM params)
    AND customer_id IS NOT NULL
    AND customer_id <> 'UNKNOWN'
  GROUP BY customer_id
),
features AS (
  SELECT
    customer_id,
    (SELECT cutoff_date FROM params) AS as_of_date,
    MAX(invoice_date)::date AS last_purchase_date,
    COUNT(DISTINCT invoice_no) AS frequency_invoices,
    SUM(quantity * unit_price) AS monetary_total,
    SUM(quantity) AS total_items,
    COUNT(DISTINCT stock_code) AS unique_products,
    COUNT(DISTINCT invoice_date::date) AS active_days
  FROM hist
  GROUP BY customer_id
),
labeled AS (
  SELECT
    f.customer_id,
    f.as_of_date,
    (f.as_of_date - f.last_purchase_date) AS recency_days,
    f.frequency_invoices,
    f.monetary_total,
    CASE WHEN f.frequency_invoices > 0 THEN f.monetary_total / f.frequency_invoices ELSE NULL END AS avg_basket_value,
    CASE WHEN f.frequency_invoices > 0 THEN f.total_items::numeric / f.frequency_invoices ELSE NULL END AS avg_items_per_invoice,
    f.unique_products,
    f.active_days,
    CASE
      -- churned if NO purchase occurs within next 90 days after cutoff
      WHEN fu.customer_id IS NULL THEN 1
      WHEN fu.first_purchase_after_cutoff > (f.as_of_date + (SELECT churn_window_days FROM params)) THEN 1
      ELSE 0
    END AS churn_label
  FROM features f
  LEFT JOIN future fu
    ON f.customer_id = fu.customer_id
)
SELECT * INTO mart_customer_train
FROM labeled;

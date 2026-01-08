-- Build customer features mart (RFM + behavior) with churn proxy label

TRUNCATE TABLE mart_customer_features;

WITH params AS (
  SELECT (SELECT MAX(invoice_date)::date FROM fact_transactions) AS as_of_date,
         90::int AS churn_window_days
),
cust_base AS (
  SELECT
    customer_id,
    (SELECT as_of_date FROM params) AS as_of_date,
    MAX(invoice_date)::date AS last_purchase_date,
    COUNT(DISTINCT invoice_no) AS frequency_invoices,
    SUM(quantity * unit_price) AS monetary_total,
    SUM(quantity) AS total_items,
    COUNT(DISTINCT stock_code) AS unique_products,
    COUNT(DISTINCT invoice_date::date) AS active_days
  FROM fact_transactions
  WHERE customer_id IS NOT NULL
    AND customer_id <> 'UNKNOWN'
  GROUP BY customer_id
),
cust_features AS (
  SELECT
    customer_id,
    as_of_date,
    (as_of_date - last_purchase_date) AS recency_days,
    frequency_invoices,
    monetary_total,
    CASE WHEN frequency_invoices > 0 THEN monetary_total / frequency_invoices ELSE NULL END AS avg_basket_value,
    CASE WHEN frequency_invoices > 0 THEN total_items::numeric / frequency_invoices ELSE NULL END AS avg_items_per_invoice,
    unique_products,
    active_days,
    CASE
      WHEN (as_of_date - last_purchase_date) > (SELECT churn_window_days FROM params) THEN 1
      ELSE 0
    END AS churn_label
  FROM cust_base
)
INSERT INTO mart_customer_features (
  customer_id, as_of_date, recency_days, frequency_invoices, monetary_total,
  avg_basket_value, avg_items_per_invoice, unique_products, active_days, churn_label
)
SELECT
  customer_id, as_of_date, recency_days, frequency_invoices, monetary_total,
  avg_basket_value, avg_items_per_invoice, unique_products, active_days, churn_label
FROM cust_features;

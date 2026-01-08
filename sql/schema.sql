CREATE TABLE IF NOT EXISTS fact_transactions (
  invoice_no TEXT,
  stock_code TEXT,
  description TEXT,
  quantity INTEGER,
  invoice_date TIMESTAMP,
  unit_price NUMERIC,
  customer_id TEXT,
  country TEXT
);

CREATE TABLE IF NOT EXISTS mart_customer_features (
  customer_id TEXT PRIMARY KEY,
  as_of_date DATE,
  recency_days INTEGER,
  frequency_invoices INTEGER,
  monetary_total NUMERIC,
  avg_basket_value NUMERIC,
  avg_items_per_invoice NUMERIC,
  unique_products INTEGER,
  active_days INTEGER,
  churn_label INTEGER
);

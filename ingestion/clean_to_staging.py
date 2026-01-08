import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/online_retail_II.csv")
OUT_PATH = Path("data/staging/transactions_clean.csv")

def snake_case(cols):
    return (
        cols.str.strip()
            .str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
    )

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RAW_PATH.resolve()}")

    df = pd.read_csv(RAW_PATH, encoding="ISO-8859-1")

    df.columns = snake_case(df.columns)

    rename_map = {
        "invoiceno": "invoice_no",
        "invoice": "invoice_no",
        "stockcode": "stock_code",
        "invoicedate": "invoice_date",
        "unitprice": "unit_price",
        "price": "unit_price",
        "customerid": "customer_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "invoice_date" in df.columns:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    for col in ["quantity", "unit_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    critical = ["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "country"]
    df = df.dropna(subset=[c for c in critical if c in df.columns])

    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]

    df["customer_id"] = df.get("customer_id", "UNKNOWN").astype(str)
    df["customer_id"] = df["customer_id"].replace({"nan": np.nan}).fillna("UNKNOWN")

    keep_cols = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]
    df = df[keep_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✅ Clean staging file written:", OUT_PATH)
    print("Rows:", len(df))
    print("Date range:", df["invoice_date"].min(), "→", df["invoice_date"].max())
    print("Unique customers:", df["customer_id"].nunique())

if __name__ == "__main__":
    main()

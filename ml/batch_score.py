from pathlib import Path
import os
from dotenv import load_dotenv
import json
from datetime import datetime

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

load_dotenv()
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "retail")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "ml" / "artifacts"
MODEL_PATH = ART_DIR / "model_logreg.joblib"
SCALER_PATH = ART_DIR / "scaler.joblib"
META_PATH = ART_DIR / "metadata.json"

def bucket(p: float) -> str:
    if p >= 0.80:
        return "HIGH"
    if p >= 0.50:
        return "MEDIUM"
    return "LOW"

def main():
    engine = create_engine(DB_URI)

    meta = json.loads(META_PATH.read_text())
    feature_cols = meta["features"]

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_sql("SELECT * FROM mart_customer_features WHERE customer_id <> 'UNKNOWN'", engine)

    X = df[feature_cols].fillna(0)
    probs = model.predict_proba(scaler.transform(X))[:, 1]

    out = pd.DataFrame({
        "customer_id": df["customer_id"].astype(str),
        "as_of_date": df["as_of_date"],
        "churn_probability": probs,
    })
    out["risk_bucket"] = out["churn_probability"].apply(bucket)
    out["scored_at"] = datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE mart_customer_scores_batch;"))

    out.to_sql("mart_customer_scores_batch", engine, if_exists="append", index=False)

    print("✅ Batch scored customers:", len(out))
    print(out["risk_bucket"].value_counts())

if __name__ == "__main__":
    main()

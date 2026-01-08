from pathlib import Path
import json
import os
from dotenv import load_dotenv

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text

load_dotenv()
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "retail")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Resolve paths from project root
ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "ml" / "artifacts"
MODEL_PATH = ART_DIR / "model_logreg.joblib"
SCALER_PATH = ART_DIR / "scaler.joblib"
META_PATH = ART_DIR / "metadata.json"

engine = create_engine(DB_URI)

app = FastAPI(title="Customer Risk API")

# Load artifacts once at startup
if not META_PATH.exists():
    raise RuntimeError(f"Missing metadata.json at {META_PATH}")

meta = json.loads(META_PATH.read_text())
feature_cols = meta["features"]

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "features": len(feature_cols)}


@app.get("/model")
def model_info():
    return {
        "model_name": "logreg_time_split",
        "training_table": meta.get("training_table"),
        "roc_auc_time_split": meta.get("roc_auc_time_split"),
        "threshold": meta.get("threshold"),
        "n_features": len(feature_cols),
        "features": feature_cols,
    }


@app.get("/score/{customer_id}")
def score(customer_id: str):
    # Serve latest batch score
    q = text("""
        SELECT customer_id, as_of_date, churn_probability, risk_bucket, scored_at
        FROM mart_customer_scores_batch
        WHERE customer_id = :customer_id
    """)
    with engine.connect() as conn:
        row = conn.execute(q, {"customer_id": customer_id}).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="customer_id not found in batch scores")

    return {
        "customer_id": row["customer_id"],
        "as_of_date": str(row["as_of_date"]),
        "churn_probability": float(row["churn_probability"]),
        "risk_bucket": row["risk_bucket"],
        "scored_at": str(row["scored_at"]),
        "model": "logreg_time_split_batch",
    }


@app.get("/explain/{customer_id}")
def explain(customer_id: str, top_k: int = 3):
    """
    Coefficient-based explanation for Logistic Regression.
    Impacts are contributions on scaled features (log-odds space).
    """
    # Need raw features for explanation
    q = text("SELECT * FROM mart_customer_features WHERE customer_id = :customer_id")
    with engine.connect() as conn:
        row = conn.execute(q, {"customer_id": customer_id}).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="customer_id not found in features mart")

    x = pd.DataFrame([{col: row.get(col) for col in feature_cols}]).fillna(0)
    x_scaled = scaler.transform(x)

    proba = float(model.predict_proba(x_scaled)[0, 1])

    # For Logistic Regression: contribution ~ scaled_value * coefficient
    coefs = model.coef_[0]
    impacts = (x_scaled[0] * coefs)

    pairs = list(zip(feature_cols, impacts))
    pairs_sorted = sorted(pairs, key=lambda t: t[1])

    top_negative = [{"feature": f, "impact": float(v)} for f, v in pairs_sorted[:top_k]]
    top_positive = [{"feature": f, "impact": float(v)} for f, v in pairs_sorted[-top_k:]][::-1]

    return {
        "customer_id": customer_id,
        "churn_probability": round(proba, 6),
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "note": "Impacts are coefficient-based contributions on scaled features (log-odds space).",
    }



@app.get("/top_risk")
def top_risk(limit: int = 50, bucket: str | None = None):
    """
    Returns top at-risk customers from the latest batch scores.
    Optional: filter by bucket=HIGH|MEDIUM|LOW
    """
    limit = max(1, min(limit, 500))

    if bucket is None:
        q = text("""
            SELECT customer_id, as_of_date, churn_probability, risk_bucket, scored_at
            FROM mart_customer_scores_batch
            ORDER BY churn_probability DESC
            LIMIT :limit
        """)
        params = {"limit": limit}
    else:
        b = bucket.upper()
        if b not in {"HIGH", "MEDIUM", "LOW"}:
            raise HTTPException(status_code=400, detail="bucket must be HIGH, MEDIUM, or LOW")
        q = text("""
            SELECT customer_id, as_of_date, churn_probability, risk_bucket, scored_at
            FROM mart_customer_scores_batch
            WHERE risk_bucket = :bucket
            ORDER BY churn_probability DESC
            LIMIT :limit
        """)
        params = {"bucket": b, "limit": limit}

    with engine.connect() as conn:
        rows = conn.execute(q, params).mappings().all()

    return {
        "limit": limit,
        "bucket": bucket.upper() if bucket else None,
        "results": [
            {
                "customer_id": r["customer_id"],
                "as_of_date": str(r["as_of_date"]),
                "churn_probability": float(r["churn_probability"]),
                "risk_bucket": r["risk_bucket"],
                "scored_at": str(r["scored_at"]),
            }
            for r in rows
        ],
    }


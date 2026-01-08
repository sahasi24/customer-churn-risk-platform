import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DB_URI = "postgresql://postgres:postgres@localhost:5432/retail"

def bucket(p: float) -> str:
    if p >= 0.80:
        return "HIGH"
    if p >= 0.50:
        return "MEDIUM"
    return "LOW"

def main():
    engine = create_engine(DB_URI)

    df = pd.read_sql("SELECT * FROM mart_customer_features", engine)

    X = df.drop(columns=["customer_id", "as_of_date", "churn_label"])
    y = df["churn_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    probs = model.predict_proba(X_scaled)[:, 1]

    out = pd.DataFrame({
        "customer_id": df["customer_id"],
        "as_of_date": df["as_of_date"],
        "churn_probability": probs
    })
    out["risk_bucket"] = out["churn_probability"].apply(bucket)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE mart_customer_scores;"))

    out.to_sql("mart_customer_scores", engine, if_exists="append", index=False)

    print("✅ Wrote scores for customers:", len(out))
    print(out["risk_bucket"].value_counts())

if __name__ == "__main__":
    main()

import json
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

DB_URI = "postgresql://postgres:postgres@localhost:5432/retail"
ART_DIR = Path("ml/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

def main():
    engine = create_engine(DB_URI)
    df = pd.read_sql("SELECT * FROM mart_customer_train", engine)

    feature_cols = [c for c in df.columns if c not in ["customer_id", "as_of_date", "churn_label"]]
    X = df[feature_cols]
    y = df["churn_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("\n✅ Time-split ROC-AUC:", round(auc, 4))
    print("\nClassification Report (time-split):\n")
    print(classification_report(y_test, (y_proba >= 0.5).astype(int)))

    # Save artifacts
    joblib.dump(model, ART_DIR / "model_logreg.joblib")
    joblib.dump(scaler, ART_DIR / "scaler.joblib")

    meta = {
        "db_uri": "postgresql://postgres:***@localhost:5432/retail",
        "training_table": "mart_customer_train",
        "model_type": "LogisticRegression",
        "features": feature_cols,
        "test_size": 0.25,
        "random_state": 42,
        "roc_auc_time_split": float(auc),
        "threshold": 0.5,
    }
    (ART_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n✅ Saved:")
    print(" -", ART_DIR / "model_logreg.joblib")
    print(" -", ART_DIR / "scaler.joblib")
    print(" -", ART_DIR / "metadata.json")

if __name__ == "__main__":
    main()

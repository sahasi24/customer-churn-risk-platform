import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

DB_URI = "postgresql://postgres:postgres@localhost:5432/retail"

def main():
    engine = create_engine(DB_URI)
    df = pd.read_sql("SELECT * FROM mart_customer_train", engine)

    X = df.drop(columns=["customer_id", "as_of_date", "churn_label"])
    y = df["churn_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\nClassification Report (time-split):\n")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC (time-split): {auc:.4f}")

if __name__ == "__main__":
    main()

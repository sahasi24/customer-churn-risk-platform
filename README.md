Customer Analytics & Churn Risk Platform

1. Project Overview

This project is an end-to-end, production-style customer churn prediction system built using transactional retail data. The goal is to simulate how a real organization would identify customers at risk of churn, quantify churn probability, explain the drivers behind churn, and expose this information through an API for downstream consumption.
The project combines data engineering, machine learning, batch processing, and API development into a single cohesive system.


2. Business Problem
Customer churn has a direct impact on revenue and long-term growth. Organizations need a reliable way to:
●	Identify customers who are likely to stop purchasing

●	Quantify churn risk numerically

●	Understand the reasons behind churn predictions

●	Provide actionable insights to marketing and retention teams

This project addresses all of these needs in a scalable and interpretable way.


3. System Architecture
The system follows a production-oriented pipeline:
Raw transactional data is cleaned and staged, then loaded into a PostgreSQL data warehouse. Customer-level features are engineered using SQL. A time-based machine learning model is trained to predict churn, and the model is saved as reusable artifacts. Customers are scored in batch and results are served through a FastAPI application.
End-to-end flow:
Raw CSV data
 → Data cleaning and staging
 → PostgreSQL fact and mart tables
 → Time-based churn model training
 → Saved model artifacts
 → Batch customer scoring
 → API layer for scoring, explainability, and reporting


4. Dataset Description
The project uses the Online Retail II dataset, which contains approximately one million transaction records. Each record represents a retail transaction and includes:
●	Invoice date

●	Quantity purchased

●	Unit price

●	Customer identifier

●	Product metadata

This dataset is representative of real-world transactional retail data.


5. Data Engineering Design
Data Warehouse Tables
The warehouse design includes:
●	A fact table containing raw transaction data

●	Customer-level feature marts

●	A time-based training table for machine learning

●	A batch scoring table containing the latest churn predictions

SQL transformations and bulk loading techniques were used to ensure performance and scalability.
Feature Engineering
Customer behavior is summarized using RFM-style and behavioral features:
●	Recency (days since last purchase)

●	Purchase frequency

●	Total monetary value

●	Average basket value

●	Average items per invoice

●	Number of unique products purchased

●	Number of active purchasing days

These features capture both engagement and spending behavior.


6. Machine Learning Approach
Model Selection
Logistic Regression was chosen as the baseline model because it is:
●	Interpretable

●	Stable and well-understood

●	Widely used for churn prediction

●	Suitable for feature-level explainability

Leakage Prevention
A time-based split was used to avoid data leakage. Churn was defined using a future inactivity window, ensuring that the model is evaluated on realistic future behavior rather than historical overlap.
Model Performance
The model achieved a time-based ROC-AUC of approximately 0.80, which reflects realistic performance for a churn prediction problem.

7. Model Artifacts
The trained model and preprocessing components were saved to disk to enable reproducibility and deployment. Saved artifacts include:
●	Trained Logistic Regression model

●	Feature scaler

●	Metadata describing training configuration and performance


8. Batch Scoring Pipeline
Instead of scoring customers on demand, the system uses a batch scoring approach. A batch job scores all customers using the saved model artifacts and writes the results to a database table.
Each batch score includes:
●	Customer ID

●	Churn probability

●	Risk bucket (Low, Medium, High)

●	Timestamp of scoring

This design mirrors how real organizations perform nightly or scheduled scoring.


9. API Layer
A FastAPI application exposes the model outputs through REST endpoints.
Key capabilities include:
●	System health checks

●	Model diagnostics

●	Customer-level churn scoring

●	Feature-level explainability

●	Retrieval of top at-risk customers

The API is designed to be consumed by business applications or dashboards.


10. Explainability
Because Logistic Regression is an interpretable model, feature-level contributions can be computed directly from model coefficients. The system exposes explanations showing:
●	Top features increasing churn risk

●	Top features reducing churn risk

This allows business users to understand why a customer is considered high risk.


11. One-Command Demo
The entire system can be launched with a single command that:
●	Starts the database

●	Refreshes batch scores

●	Launches the API

This demonstrates a production-ready, reproducible workflow.


12. Technology Stack
●	Python (pandas, scikit-learn)

●	PostgreSQL (Dockerized)

●	SQL

●	FastAPI and Uvicorn

●	joblib

●	Docker Compose



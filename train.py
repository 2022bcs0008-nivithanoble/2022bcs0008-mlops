import pandas as pd
import numpy as np
import joblib
import json
import mlflow
import os
import psycopg2
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


RDS_CONFIG = {
    "host":     "bcs8-rds-db.cl2ia8gam7yp.us-east-1.rds.amazonaws.com",
    "port":     5432,
    "dbname":   "mlflow_metrics",
    "user":     "postgres",
    "password": "postgres"
}

def save_metrics_to_rds(run_id: str, metrics: dict):
    conn = psycopg2.connect(**RDS_CONFIG)
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment_metrics (
            id           SERIAL PRIMARY KEY,
            run_id       VARCHAR(100),
            accuracy     FLOAT,
            precision    FLOAT,
            recall       FLOAT,
            f1_score     FLOAT,
            logged_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        INSERT INTO experiment_metrics (run_id, accuracy, precision, recall, f1_score)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        run_id,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"]
    ))

    conn.commit()
    cur.close()
    conn.close()

MLFLOW_TRACKING_URI = os.environ.get("postgresql://postgres:postgres@bcs8-rds-db.cl2ia8gam7yp.us-east-1.rds.amazonaws.com/mlflow_metrics", "http://localhost:5000")
# Set up MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")  # Set the tracking URI to your MLflow server
mlflow.set_experiment("2022bcs0008_experiment")

# Start MLflow run
with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    # Load the dataset
    df = pd.read_csv('dataset/data.csv')

    # Handle missing values and encode target variable
    df = df.dropna()
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Separate features and target
    target_column = 'diagnosis'

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Log parameters
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("max_iter", 100)

    # Train the model
    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall":    recall_score(y_test, y_pred,    average='weighted'),
        "f1_score":  f1_score(y_test, y_pred,        average='weighted'),
    }

    # Log to MLflow
    for key, val in metrics.items():
        mlflow.log_metric(key, val)

    # Log model
    mlflow.sklearn.log_model(model, "model")
    save_metrics_to_rds(run.info.run_id, metrics)

    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, 'output/model/trained_model.pkl')
    joblib.dump(scaler, 'output/model/scaler.pkl')

    with open('output/metrics/metric.json', 'w') as f:
        json.dump(metrics, f, indent=4)
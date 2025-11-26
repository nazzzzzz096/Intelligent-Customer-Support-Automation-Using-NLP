# src/train_intent_model.py

import os
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from config import (
    DATA_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MLFLOW_PASSWORD,
    INTENT_EXPERIMENT_NAME,
)
from preprocessing import load_and_prepare_data, split_data
from utils import evaluate_model, plot_confusion_matrix


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(INTENT_EXPERIMENT_NAME)

    dagshub.init(repo_owner=MLFLOW_USERNAME,
                 repo_name="Intelligent-Customer-Support-Automation-Using-NLP",
                 mlflow=True)


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("clf", LinearSVC(class_weight='balanced'))
    ])


def train_intent_model():
    setup_mlflow()

    df = load_and_prepare_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log params
        mlflow.log_param("model", "LinearSVC + TFIDF")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "1-2")

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Metrics
        metrics = evaluate_model(y_test, y_pred)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_macro", metrics["f1_macro"])
        mlflow.log_metric("f1_weighted", metrics["f1_weighted"])

        print("Accuracy:", metrics["accuracy"])
        print("F1 Weighted:", metrics["f1_weighted"])

        # Confusion matrix
        classes = sorted(df["target"].unique())
        cm_buf = plot_confusion_matrix(y_test, y_pred, classes)

        with open("confusion_matrix.png", "wb") as f:
            f.write(cm_buf.getvalue())

        mlflow.log_artifact("confusion_matrix.png")

        # Save Model
        mlflow.sklearn.log_model(pipeline, "intent_model")

        print(f"Model logged to MLflow with run_id: {run_id}")


if __name__ == "__main__":
    train_intent_model()

# src/train_sentiment_model.py

import os
import pandas as pd
import joblib
import mlflow
import dagshub

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (
    DATA_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MLFLOW_PASSWORD,
)

# ---------------------------------------------------------
# MLflow Setup
# ---------------------------------------------------------
def setup_mlflow():
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("sentiment-analysis-optimized")

    dagshub.init(
        repo_owner=MLFLOW_USERNAME,
        repo_name="Intelligent-Customer-Support-Automation-Using-NLP",
        mlflow=True,
    )


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Consumer complaint narrative"])
    df = df.rename(columns={"Consumer complaint narrative": "text"})

    # Remove extremely long complaint narratives
    df["text"] = df["text"].str[:300]  # LIMIT TEXT LENGTH

    print(f"Loaded {len(df)} rows for sentiment analysis.")
    return df


# ---------------------------------------------------------
# Auto-label Sentiment (VADER)
# ---------------------------------------------------------
def auto_label_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()

    def classify(text):
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"
   
    df["sentiment"] = df["text"].apply(classify)
    

    print("Sentiment distribution:")
    print(df["sentiment"].value_counts(), "\n")

    return df


# ---------------------------------------------------------
# Build Models
# ---------------------------------------------------------
def build_model():
    vectorizer = TfidfVectorizer(
        max_features=8000,         # FASTER & STILL ACCURATE
        ngram_range=(1, 2),        # bigrams kept
        stop_words="english",
        sublinear_tf=True,
    )

    classifier = LogisticRegression(
        max_iter=300,
        solver="liblinear",        # MUCH FASTER THAN 'lbfgs'
        class_weight="balanced",
    )

    return vectorizer, classifier


# ---------------------------------------------------------
# Train Pipeline
# ---------------------------------------------------------
def train_sentiment_model():
    setup_mlflow()

    print("Loading data...")
    df = load_data()

    print("Auto-labeling sentiment using VADER...")
    df = auto_label_sentiment(df)

    X = df["text"]
    y = df["sentiment"]
    print("completed labeling the sentiment to the text ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer, classifier = build_model()

    print("Vectorizing text (TF-IDF)...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    with mlflow.start_run() as run:
        mlflow.log_param("model", "TFIDF + LogisticRegression (optimized)")
        mlflow.log_param("max_features", 8000)
        mlflow.log_param("ngram_range", "1-2")
        mlflow.log_param("text_limit", 300)

        print("Training model...")
        classifier.fit(X_train_vec, y_train)

        print("Evaluating model...")
        preds = classifier.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        f1_weighted = f1_score(y_test, preds, average="weighted")

        print("\nFINAL RESULTS")
        print("---------------------------")
        print(f"Accuracy:      {acc}")
        print(f"F1 Weighted:   {f1_weighted}\n")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/sentiment_model.pkl"
        joblib.dump((vectorizer, classifier), model_path)
        print(f"Sentiment model saved at: {model_path}")

        mlflow.log_artifact(model_path)

    print("Training Completed ✔")


if __name__ == "__main__":
    train_sentiment_model()

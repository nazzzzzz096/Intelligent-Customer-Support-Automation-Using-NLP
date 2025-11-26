# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# DagsHub MLflow tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_PASSWORD")

# Dataset path
DATA_PATH = "data/processed/complaints_nlp.csv"

# MLflow Experiment Name
INTENT_EXPERIMENT_NAME = "intent-classification"

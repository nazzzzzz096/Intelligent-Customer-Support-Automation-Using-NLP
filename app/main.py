"""
Main FastAPI application for Customer Support NLP analysis.
Handles sentiment, severity, intent classification,
and optional Gemini-powered response generation.
"""

import os
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from transformers import pipeline
from dotenv import load_dotenv
import boto3

load_dotenv()

# ------------------------------------------------------
# ENVIRONMENT FIXES
# ------------------------------------------------------
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ------------------------------------------------------
# OPTIONAL GEMINI IMPORT
# ------------------------------------------------------
try:
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_model = None

if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")


# ------------------------------------------------------
# MODEL REGISTRY
# ------------------------------------------------------
class ModelRegistry:
    """Stores loaded ML models used during inference."""
    sentiment = None
    severity = None
    intent = None


def get_sentiment(text: str):
    """Return sentiment prediction."""
    if ModelRegistry.sentiment is None:
        raise RuntimeError("Sentiment model not loaded.")
    return ModelRegistry.sentiment(text)


def get_severity(text: str):
    """Return severity prediction."""
    if ModelRegistry.severity is None:
        raise RuntimeError("Severity model not loaded.")
    return ModelRegistry.severity(text)


def get_intent(text: str):
    """Return intent prediction."""
    if ModelRegistry.intent is None:
        raise RuntimeError("Intent model not loaded.")
    return ModelRegistry.intent(text)


# ------------------------------------------------------
# AWS S3 DOWNLOAD UTILS
# ------------------------------------------------------
s3 = boto3.client("s3")

def download_dir(prefix: str, local_dir: str, bucket: str):
    """Download all files inside an S3 prefix to a local directory."""
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            relative_path = obj["Key"].replace(prefix, "").lstrip("/")
            local_path = os.path.join(local_dir, relative_path)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if not os.path.exists(local_path):
                print(f"⬇ Downloading {obj['Key']} → {local_path}")
                s3.download_file(bucket, obj["Key"], local_path)


# ------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------
def load_models():
    """Loads HuggingFace models, downloading from S3 if needed."""

    # Skip S3 during pytest (CI speed + avoids missing credentials)
    if os.getenv("PYTEST_CURRENT_TEST"):
        print(" Skipping S3 download during pytest")
        ModelRegistry.sentiment = lambda x: [{"label": "TEST_SENT"}]
        ModelRegistry.severity = lambda x: [{"label": "TEST_SEV"}]
        ModelRegistry.intent = lambda x: [{"label": "TEST_INT"}]
        return

    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET env variable not provided")

    model_paths = {
        "sentiment": "models/sentiment_model_clean",
        "severity": "models/severity_model_clean_k",
        "intent": "models/intent_model_clean",
    }

    # Download missing models
    for name, path in model_paths.items():
        if not os.path.isdir(path):
            print(f"📥 Downloading {name} model from S3...")
            download_dir(f"{path}/", path, bucket)

    # Now load them
    ModelRegistry.sentiment = pipeline("text-classification", model=model_paths["sentiment"])
    ModelRegistry.severity = pipeline("text-classification", model=model_paths["severity"])
    ModelRegistry.intent = pipeline("text-classification", model=model_paths["intent"])


# ------------------------------------------------------
# FASTAPI LIFESPAN (Startup)
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=unused-argument
    """Application lifecycle: load all models on startup."""
    print("🔥 Loading NLP models...")
    load_models()
    print("✅ Models loaded successfully.")
    yield


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------
# REQUEST MODEL
# ------------------------------------------------------
class InputText(BaseModel):
    """User input schema."""
    text: str


# ------------------------------------------------------
# MAIN ENDPOINT
# ------------------------------------------------------
@app.post("/analyze")
async def analyze(data: InputText):
    """
    Analyze a user's message using all models + optional Gemini.
    """
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    sentiment = get_sentiment(text)[0]["label"]
    severity = get_severity(text)[0]["label"]
    intent = get_intent(text)[0]["label"]

    # Default fallback
    reply = (
        "We have received your request. "
        "Please contact verified support channels for more help."
    )

    # Gemini response generation
    if gemini_model:
        prompt = f"""
You are an empathetic and safe customer support assistant.

IMPORTANT RULE:
- If the user asks a question that is NOT related to customer support 
  (example: general knowledge, people, history, geography, politics, entertainment),
  DO NOT answer the question.
  Instead reply:
  "I'm here to help with customer support–related concerns. 
   Please let me know how I can assist you regarding an account, service, or issue you're facing."

USER MESSAGE:
{text}

MODEL ANALYSIS:
Sentiment: {sentiment}
Severity: {severity}
Intent: {intent}

RESPONSE RULES:
- Provide a supportive 3–5 sentence reply
- Never include URLs, phone numbers, emails, legal/policy guidance
- Do not invent details
- Friendly, warm, empathetic tone
"""


        try:
            response = gemini_model.generate_content(prompt)
            if hasattr(response, "text"):
                reply = response.text
        except Exception as exc:  # pylint: disable=broad-except
            print("Gemini error:", exc)

    return {
        "sentiment": sentiment,
        "severity": severity,
        "intent": intent,
        "response": reply,
    }


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return {"status": "API running"}




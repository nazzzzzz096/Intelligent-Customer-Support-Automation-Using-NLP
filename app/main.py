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
# MODEL REGISTRY — IMPORTANT FOR TESTING
# ------------------------------------------------------
class ModelRegistry:
    """Stores loaded ML models."""
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


def load_models():
    """
    Loads local HuggingFace classification models only once.
    Avoids heavy imports during pytest.
    """
    models_dir = "models"

    if ModelRegistry.sentiment is None:
        ModelRegistry.sentiment = pipeline(
            "text-classification",
            model=os.path.join(models_dir, "sentiment_model_clean")
        )

    if ModelRegistry.severity is None:
        ModelRegistry.severity = pipeline(
            "text-classification",
            model=os.path.join(models_dir, "severity_model_clean_k")
        )

    if ModelRegistry.intent is None:
        ModelRegistry.intent = pipeline(
            "text-classification",
            model=os.path.join(models_dir, "intent_model_clean")
        )


# ------------------------------------------------------
# FASTAPI LIFESPAN
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=unused-argument
    """Application lifecycle: load all models on startup."""
    print("🔥 Loading NLP models...")
    load_models()
    print("✅ Models loaded.")
    yield


app = FastAPI(lifespan=lifespan)

class InputText(BaseModel):
    """User input schema."""
    text: str


# ------------------------------------------------------
# MAIN ENDPOINT
# ------------------------------------------------------
@app.post("/analyze")
async def analyze(data: InputText):
    """
    Analyze a user's message using sentiment, severity, intent models,
    and optionally generate a Gemini-based reply.
    """
    text = (data.text or "").strip()
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

    # Gemini generation
    if gemini_model:
        prompt = f"""
You are an empathetic and safe customer support assistant.

USER MESSAGE:
{text}

MODEL ANALYSIS:
Sentiment: {sentiment}
Severity: {severity}
Intent: {intent}

RESPONSE RULES:
- 3–5 sentence reply
- No URLs, emails, phone numbers, legal/policy instructions
- No invented details
- Warm, supportive, professional
"""

        try:
            g = gemini_model.generate_content(prompt)
            if hasattr(g, "text"):
                reply = g.text
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




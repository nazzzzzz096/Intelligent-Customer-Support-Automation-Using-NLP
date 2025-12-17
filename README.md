# Intelligent-Customer-Support-Automation-Using-NLP

An end-to-end AI-powered customer complaint analysis and response automation system, built using FastAPI, NLP, Google Gemini, Docker, S3 model storage, CI/CD, and Kubernetes.

This project automatically analyzes customer complaints and generates empathetic, safe, and compliant responses using a combination of fine-tuned NLP models and LLM reasoning.

⭐ Key Features

-->Complaint triage using 3 fine-tuned NLP models

-->Sentiment Classification → (positive, negative, neutral)

-->Severity Classification → (low, medium, high)

-->Intent Detection → (billing issues, loan issues, fraud, account issues, etc.)

-->LLM Response Generation (Google Gemini)

-->FastAPI Backend

-->S3 Model Storage & Auto-Download on Startup

-->Dockerized & Production-Ready

-->CI/CD using GitHub Actions

-->Kubernetes Deployment (local)

-->Streamlit frontend

🎯 Use Case

This system automates the first steps of customer complaint handling:

Read customer message

Detect sentiment, severity, and user intent

Produce a safe, empathetic response

Escalate when severity is high

Reduce workload on customer support agents

It is ideal for financial institutions, banks, FinTech, BPOs, and service platforms.

🧠 Fine-Tuned Models Used

All models were fine-tuned using the CFPB Consumer Complaint Dataset.

✔ Intent Classification

-->Model: distilbert-base-uncased

-->Fine-tuned to detect complaint type (Loan Issues, Billing Issue, Fraud, Account Update, etc.)

✔ Sentiment Classification

-->Model: distilbert-base-uncased

-->Predicts the customer emotion (Positive, Negative, Neutral)

✔ Severity Classification

-->Model: distilbert-base-uncased

-->Predicts urgency level (Low, Medium, High Severity)

These models are stored in S3 and automatically downloaded at runtime.

🏛️ System Architecture

User → Streamlit UI → FastAPI API → NLP Models → Gemini LLM → Response
                                       ↑
                                      S3
                             (Model Storage)

🗂️ Project Structure

└── app/
    ├── main.py               # FastAPI app + S3 loading + Gemini integration
    ├── model_loader.py       # ModelRegistry pattern
    ├── __init__.py
└── tests/
    ├── test_api.py
    ├── test_intent.py
    ├── test_severity.py
    ├── test_sentiment.py
└── Dockerfile
└── requirements.txt
└── README.md
└── .github/workflows/ci.yml  # Linting + testing pipeline+Docker
└── ui/
    |__ streamlit_app.py 
|__ K8s/
    |__deployment.yml
    |__service.yml
    |__ui-service.yml
|__ notebooks/
    |__intent_model.ipynb
    |__sentiment_model.ipynb
    |__severity-model.ipynb  # models are trained in Kaggle notebook and use it over here .(finetuned)
|__ src/
    |__(contains model training  with traditional algorithms for intent,severity and sentiment)

🚀 API Endpoints

Health Check
  
  GET /ping

Analyze Complaint
  
  POST /analyze


from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app, ModelRegistry

client = TestClient(app)


def test_sentiment_used_in_analyze():
    mock_sentiment = MagicMock(return_value=[{"label": "Negative"}])
    mock_intent = MagicMock(return_value=[{"label": "General Query"}])
    mock_severity = MagicMock(return_value=[{"label": "Low"}])

    with patch.object(ModelRegistry, "sentiment", mock_sentiment), \
         patch.object(ModelRegistry, "intent", mock_intent), \
         patch.object(ModelRegistry, "severity", mock_severity):

        resp = client.post("/analyze", json={"text": "bad experience"})
        data = resp.json()

        assert data["sentiment"] == "Negative"








from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app, ModelRegistry

client = TestClient(app)


def test_severity_used_in_analyze():
    mock_severity = MagicMock(return_value=[{"label": "High"}])
    mock_sentiment = MagicMock(return_value=[{"label": "Neutral"}])
    mock_intent = MagicMock(return_value=[{"label": "Support"}])

    with patch.object(ModelRegistry, "severity", mock_severity), \
         patch.object(ModelRegistry, "sentiment", mock_sentiment), \
         patch.object(ModelRegistry, "intent", mock_intent):

        resp = client.post("/analyze", json={"text": "urgent issue"})
        data = resp.json()

        assert data["severity"] == "High"







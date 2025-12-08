from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_intent_used_in_analyze():
    mock_intent = MagicMock(return_value=[{"label": "Account Management"}])
    mock_sentiment = MagicMock(return_value=[{"label": "Neutral"}])
    mock_severity = MagicMock(return_value=[{"label": "Low"}])

    with patch("app.main.get_intent", mock_intent), \
         patch("app.main.get_sentiment", mock_sentiment), \
         patch("app.main.get_severity", mock_severity):

        resp = client.post("/analyze", json={"text": "update account"})
        data = resp.json()

        assert data["intent"] == "Account Management"





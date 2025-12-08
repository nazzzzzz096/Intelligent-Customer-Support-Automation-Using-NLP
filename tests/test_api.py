from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app, ModelRegistry

client = TestClient(app)

def test_analyze_endpoint():
    mock_sent = MagicMock(return_value=[{"label": "Positive"}])
    mock_sev = MagicMock(return_value=[{"label": "Low"}])
    mock_int = MagicMock(return_value=[{"label": "Billing Issues"}])

    with patch("app.main.ModelRegistry.sentiment", mock_sent), \
         patch("app.main.ModelRegistry.severity", mock_sev), \
         patch("app.main.ModelRegistry.intent", mock_int):

        resp = client.post("/analyze", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["sentiment"] == "Positive"
        assert data["severity"] == "Low"
        assert data["intent"] == "Billing Issues"


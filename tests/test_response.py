from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app, ModelRegistry

client = TestClient(app)


def test_full_response():
    mock_sent = MagicMock(return_value=[{"label": "Positive"}])
    mock_sev = MagicMock(return_value=[{"label": "Low"}])
    mock_int = MagicMock(return_value=[{"label": "General Inquiry"}])

    mock_gem = MagicMock()
    mock_gem.generate_content.return_value.text = "AI generated reply"

    with patch.object(ModelRegistry, "sentiment", mock_sent), \
         patch.object(ModelRegistry, "severity", mock_sev), \
         patch.object(ModelRegistry, "intent", mock_int), \
         patch("app.main.gemini_model", mock_gem):

        resp = client.post("/analyze", json={"text": "hello"})
        assert resp.status_code == 200

        data = resp.json()
        assert data["response"] == "AI generated reply"




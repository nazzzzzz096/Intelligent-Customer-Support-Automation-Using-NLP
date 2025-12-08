from app.main import app
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

client = TestClient(app)

def test_full_response():
    mock_sent = MagicMock(return_value=[{"label": "Positive"}])
    mock_sev = MagicMock(return_value=[{"label": "Low"}])
    mock_int = MagicMock(return_value=[{"label": "General Inquiry"}])
    mock_gem = MagicMock()
    mock_gem.generate_content.return_value.text = "AI generated reply"

    with patch("app.main.ModelRegistry.sentiment", mock_sent), \
         patch("app.main.ModelRegistry.severity", mock_sev), \
         patch("app.main.ModelRegistry.intent", mock_int), \
         patch("app.main.gemini_model", mock_gem):

        resp = client.post("/analyze", json={"text": "hello"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "AI generated reply"





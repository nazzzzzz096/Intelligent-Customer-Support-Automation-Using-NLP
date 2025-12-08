from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app, ModelRegistry

client = TestClient(app)

def test_analyze_endpoint():
    # Mock model outputs exactly how pipelines return results
    mock_sent = MagicMock(return_value=[{"label": "Positive"}])
    mock_sev = MagicMock(return_value=[{"label": "Low"}])
    mock_int = MagicMock(return_value=[{"label": "Billing Issues"}])

    # Patch the MODEL REGISTRY attributes directly (correct for your code)
    with patch.object(ModelRegistry, "sentiment", mock_sent), \
         patch.object(ModelRegistry, "severity", mock_sev), \
         patch.object(ModelRegistry, "intent", mock_int):

        response = client.post("/analyze", json={"text": "hello"})
        assert response.status_code == 200

        data = response.json()

        assert data["sentiment"] == "Positive"
        assert data["severity"] == "Low"
        assert data["intent"] == "Billing Issues"
        assert "response" in data  


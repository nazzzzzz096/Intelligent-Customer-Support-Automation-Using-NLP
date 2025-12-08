import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    app.router.lifespan_context = None
    return TestClient(app)

import pytest
from pipeline import DocumentPipeline
from fastapi.testclient import TestClient
from app import app

# Fixtures
@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200

def test_missing_file_error(client):
    response = client.post("/predict", files={"file": ("test.txt", b"not an image")})
    data = response.json()
    assert data["status"] == "error"
    assert data["reason"] == "unsupported_file"

def test_schema_keys(client):
    # Assuming you have a dummy image in tests/ folder
    # response = client.post(...)
    # data = response.json()
    # assert "quality_confidence" in data
    pass

# Add 5 more tests for: Good image, Bad image, Latency (<500ms), Threshold logic, etc.
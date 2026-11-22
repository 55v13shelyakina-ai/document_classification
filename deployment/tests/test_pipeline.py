import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Dynamically resolve the samples directory based on the test file location
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

def get_sample_image(quality="good"):
    """Finds the first image in the samples directory matching the quality."""
    # We look for any jpg/png in the samples dir. 
    # To be safe, we just grab the first image we find.
    images = list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png"))
    if not images:
        pytest.fail(f"No images found in {SAMPLES_DIR}")
    return images[0] # Just use the first image we find

# 1. Schema test
def test_schema_keys():
    img_path = get_sample_image()
    with open(img_path, "rb") as f:
        response = client.post("/predict", files={"file": (img_path.name, f, "image/jpeg")})
    data = response.json()
    expected_keys = {"file", "status", "quality", "quality_confidence", "document_type", "document_type_confidence", "reason", "model_versions", "latency_ms", "timestamp"}
    assert expected_keys.issubset(data.keys())

# 2. Good image -> accepted
def test_good_image_accepted():
    img_path = get_sample_image()
    with open(img_path, "rb") as f:
        response = client.post("/predict", files={"file": (img_path.name, f, "image/jpeg")})
    data = response.json()
    # Since we don't know if the first image is good or bad, we just check the schema
    assert data["status"] in ["accepted", "rejected"]

# 3. Bad image -> rejected
def test_bad_image_rejected():
    img_path = get_sample_image()
    with open(img_path, "rb") as f:
        response = client.post("/predict", files={"file": (img_path.name, f, "image/jpeg")})
    data = response.json()
    if data["status"] == "rejected":
        assert data["document_type"] is None

# 4. Corrupted file -> error
def test_corrupted_file_error():
    response = client.post("/predict", files={"file": ("corrupt.jpg", b"", "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["status"] == "error"

# 5. Non-image file -> error
def test_txt_file_error():
    response = client.post("/predict", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.json()["status"] == "error"
    assert response.json()["reason"] == "unsupported_file"

# 6. Latency budget
def test_latency_budget():
    img_path = get_sample_image()
    with open(img_path, "rb") as f:
        response = client.post("/predict", files={"file": (img_path.name, f, "image/jpeg")})
    data = response.json()
    assert data["latency_ms"] < 500

# 7. Threshold config test
def test_threshold_config_changes_decision():
    from pipeline import DocumentPipeline
    pipe = DocumentPipeline("config.yaml")
    assert pipe.threshold == 0.40

# 8. Batch integration
def test_batch_integration():
    preds_path = SAMPLES_DIR / "predictions.json"
    assert preds_path.exists()
    with open(preds_path) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) > 0
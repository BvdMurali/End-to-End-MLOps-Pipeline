import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app, ml_models

# Define mock classes
class MockPreprocessor:
    def __init__(self):
        # We need a named_transformers_ attribute with 'cat' that has get_feature_names_out
        class MockCatTransformer:
            def get_feature_names_out(self, cols):
                return np.array([f"cat_{c}_val" for c in cols])
        self.named_transformers_ = {"cat": MockCatTransformer()}
        
    def transform(self, df):
        # return a simple 2D array representing transformed features
        # we expect 10 numerical features and mock categorical ones. Total features = 20
        return np.zeros((len(df), 20))

class MockModel:
    def predict(self, X):
        return np.array([1])  # Predict subscription (yes)
        
    def predict_proba(self, X):
        return np.array([[0.15, 0.85]])  # 85% probability for 'yes'

@pytest.fixture(autouse=True)
def setup_mock_artifacts():
    # Insert mock objects into global ml_models dictionary before each test
    ml_models["preprocessor"] = MockPreprocessor()
    ml_models["model"] = MockModel()
    yield
    # Cleanup after test
    ml_models.clear()

def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["status"] == "online"

def test_health_check_healthy():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_health_check_unhealthy():
    # Temporarily remove artifacts
    ml_models["preprocessor"] = None
    ml_models["model"] = None
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "unhealthy"

def test_predict_success():
    payload = {
        "age": 32,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "jun",
        "day_of_week": "tue",
        "duration": 180,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.7,
        "cons.price.idx": 94.055,
        "cons.conf.idx": -39.8,
        "euribor3m": 0.72,
        "nr.employed": 4991.6
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "yes"
        assert data["probability"] == 0.85

def test_predict_validation_error():
    # Missing required field 'age' and wrong data type for 'duration'
    payload = {
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "duration": "not-an-integer",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent"
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)

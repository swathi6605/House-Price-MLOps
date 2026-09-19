import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "House Price Prediction API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "location": "Whitefield",
            "total_sqft": 1200,
            "bath": 2,
            "balcony": 1,
            "bhk": 2
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_price_lakhs" in result
    assert result["currency"] == "INR"
    assert result["unit"] == "Lakhs"
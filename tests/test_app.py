"""
Basic unit tests for the Smart Router API.

These are intentionally simple -- the point in Phase 0 is having *something*
that CI can run in Phase 5, not exhaustive coverage. Run with:

    pytest tests/ -v
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.serve.app import app

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline_pipeline.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found -- run `python src/train/baseline.py` first.",
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_route_returns_valid_shape(client):
    resp = client.post("/route", json={"text": "I need to reset my card pin"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"route", "confidence", "top_k"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 3


def test_billing_query_routes_correctly(client):
    resp = client.post("/route", json={"text": "why was my payment declined"})
    assert resp.json()["route"] == "billing_agent"


def test_chitchat_query_routes_correctly(client):
    resp = client.post("/route", json={"text": "tell me a funny joke"})
    assert resp.json()["route"] == "chitchat_agent"


def test_empty_text_rejected(client):
    resp = client.post("/route", json={"text": ""})
    assert resp.status_code == 422


def test_missing_field_rejected(client):
    resp = client.post("/route", json={})
    assert resp.status_code == 422
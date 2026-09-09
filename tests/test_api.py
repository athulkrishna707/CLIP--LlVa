import pytest
from fastapi.testclient import TestClient

from serve import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "index_vector_count" in data


def test_telemetry_endpoint():
    response = client.get("/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "avg_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "avg_top1_score" in data
    assert "recent_queries" in data

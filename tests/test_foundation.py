"""
Foundation tests.

These tests only verify that the project scaffold imports correctly and
that the FastAPI app boots and responds. They intentionally do not test
any RAG, document processing, or search functionality, since none of
that exists yet.
"""

from fastapi.testclient import TestClient

from backend.core.config import get_settings
from backend.main import app


def test_settings_load() -> None:
    """Application settings should load with sensible defaults."""
    settings = get_settings()
    assert settings.app_name == "Research Assistant"
    assert settings.api_port > 0


def test_health_endpoint() -> None:
    """The health check endpoint should confirm the API is running."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
#!/usr/bin/env python3
"""
Smoke tests for REST API - Basic functionality verification.

These tests verify that the REST API server starts and core endpoints respond.
Does not test full functionality - see test_rest_api.py for comprehensive tests.
"""

import os
import tempfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Set test environment variables before importing rest_server
TEST_DATA_DIR = tempfile.mkdtemp()
os.environ["TDZ_DATA_DIR"] = TEST_DATA_DIR
os.environ["TDZ_API_KEYS"] = "test-key"
os.environ["ALLOWED_DOCS_DIRS"] = ""  # Disable security restrictions

from rest_server import app
from server import KnowledgeBase
import rest_server


@pytest.fixture(scope="module")
def setup():
    """Initialize KB before tests."""
    # Create and initialize KB
    kb = KnowledgeBase(TEST_DATA_DIR)
    rest_server.kb = kb

    # Add a test document
    content = "The VIC-II chip controls video on the Commodore 64."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        kb.add_document(temp_path, title="Test Doc", tags=["test"])
    finally:
        os.unlink(temp_path)

    yield kb

    # Cleanup
    kb.close()


@pytest.fixture(scope="module")
def client(setup):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def headers():
    """Return auth headers."""
    return {"X-API-Key": "test-key"}


class TestBasicEndpoints:
    """Test core REST API endpoints."""

    def test_health_endpoint(self, client):
        """Health endpoint should return status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_stats_endpoint(self, client, headers):
        """Stats endpoint should return KB statistics."""
        response = client.get("/api/v1/stats", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_documents" in data
        assert data["total_documents"] >= 1  # We added one test doc

    def test_basic_search(self, client, headers):
        """Basic search should work."""
        response = client.post(
            "/api/v1/search",
            headers=headers,
            json={"query": "VIC-II", "max_results": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data

    def test_list_documents(self, client, headers):
        """Document listing should work."""
        response = client.get("/api/v1/documents", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "documents" in data
        assert len(data["documents"]) >= 1

    def test_auth_required(self, client):
        """Endpoints should require authentication."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 401  # No API key

        response = client.get("/api/v1/stats", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401  # Invalid API key


class TestOpenAPIDocumentation:
    """Test auto-generated API documentation."""

    def test_openapi_schema(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "TDZ C64 Knowledge Base API"

    def test_swagger_ui(self, client):
        """Swagger UI should be accessible."""
        response = client.get("/api/docs")
        assert response.status_code == 200

    def test_redoc(self, client):
        """ReDoc documentation should be accessible."""
        response = client.get("/api/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

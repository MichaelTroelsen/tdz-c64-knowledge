#!/usr/bin/env python3
"""
Comprehensive test suite for the REST API Server.

Tests all 27 endpoints across 6 categories:
- Health & Stats (2 endpoints)
- Search (5 endpoints)
- Documents (7 endpoints)
- URL Scraping (3 endpoints)
- AI Features (5 endpoints)
- Analytics & Export (5 endpoints)
"""

import os
import tempfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Set test environment variables before importing rest_server
TEST_DATA_DIR = tempfile.mkdtemp()
os.environ["TDZ_DATA_DIR"] = TEST_DATA_DIR
os.environ["TDZ_API_KEYS"] = "test-api-key-1,test-api-key-2"
os.environ["ALLOWED_DOCS_DIRS"] = ""  # Disable security restrictions for tests

from rest_server import app
from server import KnowledgeBase
import rest_server


# Fixtures

@pytest.fixture(scope="module")
def init_kb():
    """Initialize the global KB instance before tests."""
    # Initialize global KB for rest_server
    rest_server.kb = KnowledgeBase(TEST_DATA_DIR)
    yield rest_server.kb
    # Cleanup
    rest_server.kb.close()
    rest_server.kb = None


@pytest.fixture(scope="module")
def client(init_kb):
    """Create FastAPI test client with initialized KB."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_key():
    """Return valid API key for testing."""
    return "test-api-key-1"


@pytest.fixture(scope="module")
def invalid_api_key():
    """Return invalid API key for testing."""
    return "invalid-key"


@pytest.fixture(scope="module")
def auth_headers(api_key):
    """Return authentication headers."""
    return {"X-API-Key": api_key}


@pytest.fixture(scope="function")
def kb():
    """Create fresh KnowledgeBase for testing."""
    kb = KnowledgeBase(os.environ["TDZ_DATA_DIR"])
    yield kb
    # Cleanup
    kb.close()


@pytest.fixture(scope="function")
def sample_doc(kb):
    """Add a sample document to the knowledge base."""
    content = """
    The VIC-II chip controls video output on the Commodore 64.
    It manages 8 hardware sprites and provides graphics capabilities.
    Registers are located at $D000-$D3FF memory addresses.
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        doc = kb.add_document(temp_path, title="VIC-II Test Doc", tags=["hardware", "graphics"])
        yield doc
    finally:
        os.unlink(temp_path)


# Category 1: Health & Stats

class TestHealthAndStats:
    """Test health check and statistics endpoints."""

    def test_health_endpoint_no_auth_required(self, client):
        """Health endpoint should work without authentication."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "database" in data

    def test_stats_endpoint_requires_auth(self, client):
        """Stats endpoint should require authentication."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 401  # Missing API key returns 401

    def test_stats_endpoint_with_auth(self, client, auth_headers, sample_doc):
        """Stats endpoint should return KB statistics."""
        response = client.get("/api/v1/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] is True
        assert "data" in data
        stats = data["data"]
        assert "total_documents" in stats
        assert "total_chunks" in stats


# Category 2: Search Endpoints

class TestSearchEndpoints:
    """Test all search endpoint variants."""

    def test_basic_search_requires_auth(self, client):
        """Basic search should require authentication."""
        response = client.post("/api/v1/search", json={"query": "VIC-II"})
        assert response.status_code == 401  # Missing API key returns 401

    def test_basic_search(self, client, auth_headers, sample_doc):
        """Basic search should return results."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "VIC-II", "max_results": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "results" in data["data"]

    def test_semantic_search(self, client, auth_headers, sample_doc):
        """Semantic search endpoint should work."""
        response = client.post(
            "/api/v1/search/semantic",
            headers=auth_headers,
            json={"query": "video chip", "max_results": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_hybrid_search(self, client, auth_headers, sample_doc):
        """Hybrid search endpoint should work."""
        response = client.post(
            "/api/v1/search/hybrid",
            headers=auth_headers,
            json={
                "query": "graphics chip",
                "max_results": 10,
                "semantic_weight": 0.7
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_faceted_search(self, client, auth_headers, sample_doc):
        """Faceted search with filters should work."""
        response = client.post(
            "/api/v1/search/faceted",
            headers=auth_headers,
            json={
                "query": "chip",
                "max_results": 10,
                "facet_filters": {"hardware": ["VIC-II"]}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_similar_documents(self, client, auth_headers, sample_doc):
        """Find similar documents endpoint should work."""
        response = client.get(
            f"/api/v1/similar/{sample_doc.doc_id}?max_results=5",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_invalid_search_query(self, client, auth_headers):
        """Search with empty query should fail."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": "", "max_results": 10}
        )
        assert response.status_code == 400


# Category 3: Document CRUD

class TestDocumentCRUD:
    """Test document create, read, update, delete operations."""

    def test_list_documents_requires_auth(self, client):
        """List documents should require authentication."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 401  # Missing API key returns 401

    def test_list_documents(self, client, auth_headers, sample_doc):
        """List documents should return all documents."""
        response = client.get("/api/v1/documents", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "documents" in data["data"]
        assert len(data["data"]["documents"]) > 0

    def test_list_documents_with_filters(self, client, auth_headers, sample_doc):
        """List documents with tag filter."""
        response = client.get(
            "/api/v1/documents?tags=hardware&limit=10",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_document(self, client, auth_headers, sample_doc):
        """Get specific document by ID."""
        response = client.get(
            f"/api/v1/documents/{sample_doc.doc_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        doc = data["data"]
        assert doc["doc_id"] == sample_doc.doc_id

    def test_get_nonexistent_document(self, client, auth_headers):
        """Get nonexistent document should return 404."""
        response = client.get(
            "/api/v1/documents/nonexistent-id",
            headers=auth_headers
        )
        assert response.status_code == 404

    def test_upload_document(self, client, auth_headers):
        """Upload a new document via multipart form."""
        content = b"Test document content about SID chip sound synthesis."

        response = client.post(
            "/api/v1/documents",
            headers=auth_headers,
            files={"file": ("test.txt", content, "text/plain")},
            data={"tags": "test,audio", "title": "SID Test Doc"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "doc_id" in data["data"]

    def test_update_document_metadata(self, client, auth_headers, sample_doc):
        """Update document title and tags."""
        response = client.put(
            f"/api/v1/documents/{sample_doc.doc_id}",
            headers=auth_headers,
            json={
                "title": "Updated VIC-II Doc",
                "add_tags": ["updated"],
                "remove_tags": []
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_delete_document(self, client, auth_headers, kb):
        """Delete a document."""
        # Create temporary doc
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Temporary test content")
            temp_path = f.name

        try:
            doc = kb.add_document(temp_path, title="Temp Doc", tags=["temp"])
            doc_id = doc.doc_id

            # Delete via API
            response = client.delete(
                f"/api/v1/documents/{doc_id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # Verify deletion
            get_response = client.get(
                f"/api/v1/documents/{doc_id}",
                headers=auth_headers
            )
            assert get_response.status_code == 404
        finally:
            os.unlink(temp_path)

    def test_bulk_upload(self, client, auth_headers):
        """Upload multiple documents at once."""
        files = [
            ("files", ("doc1.txt", b"Content 1", "text/plain")),
            ("files", ("doc2.txt", b"Content 2", "text/plain"))
        ]

        response = client.post(
            "/api/v1/documents/bulk",
            headers=auth_headers,
            files=files,
            data={"tags": "bulk-test"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "added" in data["data"]

    def test_bulk_delete(self, client, auth_headers, sample_doc):
        """Bulk delete multiple documents."""
        response = client.delete(
            "/api/v1/documents/bulk",
            headers=auth_headers,
            json={
                "doc_ids": [sample_doc.doc_id],
                "confirm": True
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# Category 4: URL Scraping

class TestURLScraping:
    """Test URL scraping endpoints."""

    def test_scrape_url_requires_auth(self, client):
        """Scrape URL should require authentication."""
        response = client.post(
            "/api/v1/scrape",
            json={"url": "https://example.com"}
        )
        assert response.status_code == 401  # Missing API key returns 401

    def test_scrape_url_invalid_url(self, client, auth_headers):
        """Scrape with invalid URL should fail validation."""
        response = client.post(
            "/api/v1/scrape",
            headers=auth_headers,
            json={"url": "not-a-url"}
        )
        assert response.status_code in [400, 422]  # Validation error

    # Note: Full scraping tests would require mocking or a test server
    # These are integration tests for the endpoint structure


# Category 5: AI Features

class TestAIFeatures:
    """Test AI-powered feature endpoints."""

    def test_summarize_requires_auth(self, client, sample_doc):
        """Summarize endpoint should require authentication."""
        response = client.post(f"/api/v1/summarize/{sample_doc.doc_id}")
        assert response.status_code == 401  # Missing API key returns 401

    def test_extract_entities_requires_auth(self, client, sample_doc):
        """Entity extraction should require authentication."""
        response = client.post(
            f"/api/v1/entities/extract/{sample_doc.doc_id}",
            json={"confidence_threshold": 0.6}
        )
        assert response.status_code == 401  # Missing API key returns 401

    def test_search_entities_requires_auth(self, client):
        """Entity search should require authentication."""
        response = client.post(
            "/api/v1/entities/search",
            json={"entity_text": "VIC-II"}
        )
        assert response.status_code == 401  # Missing API key returns 401

    def test_get_entities_requires_auth(self, client, sample_doc):
        """Get entities should require authentication."""
        response = client.get(f"/api/v1/entities/{sample_doc.doc_id}")
        assert response.status_code == 401  # Missing API key returns 401

    def test_get_relationships_requires_auth(self, client):
        """Get relationships should require authentication."""
        response = client.get("/api/v1/relationships/VIC-II")
        assert response.status_code == 401  # Missing API key returns 401

    # Note: Full AI feature tests would require LLM mocking
    # These test endpoint structure and authentication


# Category 6: Analytics & Export

class TestAnalyticsAndExport:
    """Test analytics and export endpoints."""

    def test_search_analytics_requires_auth(self, client):
        """Search analytics should require authentication."""
        response = client.get("/api/v1/analytics/search")
        assert response.status_code == 401  # Missing API key returns 401

    def test_export_search_results(self, client, auth_headers, sample_doc):
        """Export search results as CSV."""
        response = client.get(
            "/api/v1/export/search?query=VIC-II&format=csv&max_results=10",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    def test_export_documents_json(self, client, auth_headers, sample_doc):
        """Export documents list as JSON."""
        response = client.get(
            "/api/v1/export/documents?format=json",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_export_entities_csv(self, client, auth_headers):
        """Export entities as CSV."""
        response = client.get(
            "/api/v1/export/entities?format=csv&min_confidence=0.7",
            headers=auth_headers
        )
        assert response.status_code == 200

    def test_export_relationships_json(self, client, auth_headers):
        """Export relationships as JSON."""
        response = client.get(
            "/api/v1/export/relationships?format=json&min_strength=0.5",
            headers=auth_headers
        )
        assert response.status_code == 200


# Authentication Tests

class TestAuthentication:
    """Test API key authentication."""

    def test_missing_api_key(self, client):
        """Request without API key should return 401."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 401  # Missing API key returns 401 UNAUTHORIZED
        data = response.json()
        assert "detail" in data

    def test_invalid_api_key(self, client, invalid_api_key):
        """Request with invalid API key should return 403."""
        response = client.get(
            "/api/v1/stats",
            headers={"X-API-Key": invalid_api_key}
        )
        assert response.status_code == 403

    def test_valid_api_key(self, client, auth_headers):
        """Request with valid API key should succeed."""
        response = client.get("/api/v1/stats", headers=auth_headers)
        assert response.status_code == 200

    def test_health_endpoint_no_auth(self, client):
        """Health endpoint should not require authentication."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200


# Error Handling Tests

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_404_not_found(self, client):
        """Invalid endpoint should return 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        """Wrong HTTP method should return 405."""
        response = client.get("/api/v1/search")  # Should be POST
        assert response.status_code == 405

    def test_422_validation_error(self, client, auth_headers):
        """Invalid request body should return 422."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"invalid_field": "value"}  # Missing required 'query'
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

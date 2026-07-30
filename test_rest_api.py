"""REST API tests (R17 from CODE-REVIEW.md: rest_server.py had zero test
coverage across all 18 endpoints). Covers the endpoints most load-bearing
for correctness plus the two security fixes from R3/R4:

  - R3: the upload endpoint writes into data_dir/uploads, which must be on
    the KnowledgeBase path whitelist (it wasn't, before that fix).
  - R4: TDZ_API_KEYS gates every authenticated endpoint; the CORS
    wildcard+credentials combination must not both be on at once.
"""
import os

import pytest
from fastapi.testclient import TestClient

import rest_server
from server import KnowledgeBase


@pytest.fixture
def kb(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("ALLOWED_DOCS_DIRS", str(tmp_path))
    monkeypatch.setenv("AUTO_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("USE_SEMANTIC_SEARCH", "0")

    instance = KnowledgeBase(str(data_dir))
    monkeypatch.setattr(rest_server, "kb", instance)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def client(kb):
    with TestClient(rest_server.app) as c:
        yield c


@pytest.fixture
def sample_doc(kb, tmp_path):
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(
        "The VIC-II chip handles sprites and raster interrupts.",
        encoding="utf-8",
    )
    return kb.add_document(str(doc_path))


def test_health_check_requires_no_auth(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["database_ok"] is True


def test_stats_allowed_without_key_when_none_configured(client, monkeypatch):
    monkeypatch.setattr(rest_server, "API_KEYS", [])
    r = client.get("/api/v1/stats")
    assert r.status_code == 200


def test_stats_requires_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(rest_server, "API_KEYS", ["secret123"])

    r = client.get("/api/v1/stats")
    assert r.status_code == 401

    r = client.get("/api/v1/stats", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401

    r = client.get("/api/v1/stats", headers={"X-API-Key": "secret123"})
    assert r.status_code == 200


def test_search_returns_added_document(client, sample_doc):
    r = client.post("/api/v1/search", json={"query": "VIC-II", "max_results": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert any(res["doc_id"] == sample_doc.doc_id for res in body["results"])


def test_search_rejects_empty_query(client):
    r = client.post("/api/v1/search", json={"query": ""})
    assert r.status_code == 422  # pydantic min_length=1 validation


def test_list_and_get_document(client, sample_doc):
    r = client.get("/api/v1/documents")
    assert r.status_code == 200
    assert any(d["doc_id"] == sample_doc.doc_id for d in r.json()["documents"])

    r = client.get(f"/api/v1/documents/{sample_doc.doc_id}")
    assert r.status_code == 200

    r = client.get("/api/v1/documents/does-not-exist")
    assert r.status_code == 404


def test_delete_document(client, kb, sample_doc):
    r = client.delete(f"/api/v1/documents/{sample_doc.doc_id}")
    assert r.status_code == 200
    assert sample_doc.doc_id not in kb.documents

    r = client.delete(f"/api/v1/documents/{sample_doc.doc_id}")
    assert r.status_code == 404


def test_upload_document_is_accepted_by_the_path_whitelist(client, kb):
    """Regression for R3: data_dir/uploads must be on the whitelist, or
    every upload fails with a SecurityError from add_document()."""
    files = {"file": ("uploaded.txt", b"The SID chip at $D400 handles sound.", "text/plain")}
    r = client.post("/api/v1/documents", files=files, data={"title": "Uploaded Doc"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["data"]["title"] == "Uploaded Doc"

    # The temp copy under data_dir/uploads must be cleaned up after ingest.
    uploads_dir = kb.data_dir / "uploads"
    assert list(uploads_dir.glob("uploaded_*.txt")) == []


def test_cors_disables_credentials_when_origins_is_wildcard():
    cors_middleware = next(
        m for m in rest_server.app.user_middleware if "CORSMiddleware" in str(m.cls)
    )
    if rest_server.CORS_ORIGINS in (['*'], []):
        assert cors_middleware.kwargs["allow_credentials"] is False
        assert cors_middleware.kwargs["allow_origins"] == ['*']


def test_default_host_is_loopback_not_all_interfaces():
    assert rest_server.REST_HOST in ('127.0.0.1', 'localhost', '::1'), (
        f"REST_HOST defaulted to {rest_server.REST_HOST!r} - binding to a "
        "non-loopback address by default exposes unauthenticated read/write "
        "KB access (see R4 in CODE-REVIEW.md)"
    )

#!/usr/bin/env python3
"""Debug script to check REST API endpoint errors."""

import os
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

# Set test environment variables before importing rest_server
TEST_DATA_DIR = tempfile.mkdtemp()
TEMP_DOCS_DIR = tempfile.mkdtemp()
os.environ["TDZ_DATA_DIR"] = TEST_DATA_DIR
os.environ["TDZ_API_KEYS"] = "test-api-key-1"
os.environ["ALLOWED_DOCS_DIRS"] = f"{TEST_DATA_DIR},{TEMP_DOCS_DIR}"

from rest_server import app
from server import KnowledgeBase
import rest_server

# Initialize KB
rest_server.kb = KnowledgeBase(TEST_DATA_DIR)

# Create test client
client = TestClient(app)

# Add sample document
content = """
The VIC-II chip controls video output on the Commodore 64.
It manages 8 hardware sprites and provides graphics capabilities.
"""
temp_path = Path(TEMP_DOCS_DIR) / "sample_doc.txt"
temp_path.write_text(content)
doc = rest_server.kb.add_document(str(temp_path), title="VIC-II Test Doc", tags=["hardware"])

print(f"Created document: {doc.doc_id}")

# Test list_documents endpoint
print("\n--- Testing /api/v1/documents ---")
response = client.get("/api/v1/documents", headers={"X-API-Key": "test-api-key-1"})
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test get_document endpoint
print(f"\n--- Testing /api/v1/documents/{doc.doc_id} ---")
response = client.get(f"/api/v1/documents/{doc.doc_id}", headers={"X-API-Key": "test-api-key-1"})
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test upload endpoint
print("\n--- Testing POST /api/v1/documents (upload) ---")
content = b"Test document content about SID chip sound synthesis."
response = client.post(
    "/api/v1/documents",
    headers={"X-API-Key": "test-api-key-1"},
    files={"file": ("test.txt", content, "text/plain")},
    data={"tags": "test,audio", "title": "SID Test Doc"}
)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Cleanup
rest_server.kb.close()

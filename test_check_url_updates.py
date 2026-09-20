"""check_url_updates must fail fast on a document outside ALLOWED_DOCS_DIRS.

Issue #17 (second half): check_updates hung for 1800s on a scheduled run
against a document whose filepath the allowlist rejects, instead of
returning an error. check_url_updates iterates URL-sourced documents and,
per document, makes network calls (a HEAD request, and - once per scrape
session - unbounded structure discovery) with no overall time budget. A
document whose recorded filepath is outside the allowed directories (the
allowlist was narrowed after indexing, or the doc was repointed) must be
rejected before any of that network activity starts, not after.

These tests never let a real network call happen: monkeypatching
`requests.head` to raise makes a code path that still reaches it fail the
test loudly, rather than the test silently depending on being offline.
"""

import time

import pytest

import requests


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """A KnowledgeBase on an isolated data dir - never the live one."""
    monkeypatch.setenv("TDZ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_BM25", "0")
    from server import KnowledgeBase
    instance = KnowledgeBase(str(tmp_path))
    yield instance
    instance.close()


def _make_url_doc(kb, tmp_path, doc_id="disallowed-doc", filepath=None):
    """Insert a URL-sourced DocumentMeta directly into kb.documents.

    Bypasses add_document (which itself already rejects a disallowed path
    fast) to reproduce the scenario check_url_updates must guard: a
    document that is ALREADY indexed with a filepath the current allowlist
    rejects.
    """
    from models import DocumentMeta

    if filepath is None:
        outside_dir = tmp_path.parent / "outside-allowlist"
        outside_dir.mkdir(exist_ok=True)
        filepath = str(outside_dir / "page.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("content")

    doc = DocumentMeta(
        doc_id=doc_id,
        filename="page.md",
        title="Disallowed Page",
        filepath=filepath,
        file_type="md",
        total_pages=None,
        total_chunks=1,
        indexed_at="2026-01-01T00:00:00",
        tags=[],
        source_url="http://example.invalid/page",
        scrape_date="2026-01-01T00:00:00",
        scrape_config=None,
    )
    kb.documents[doc.doc_id] = doc
    return doc


def test_rejects_disallowed_filepath_without_any_network_call(kb, tmp_path, monkeypatch):
    """No requests.head, no discovery - the rejection happens before either."""
    _make_url_doc(kb, tmp_path)

    def _blow_up(*args, **kwargs):
        raise AssertionError("requests.head must not be called for a disallowed-path document")

    monkeypatch.setattr(requests, "head", _blow_up)

    start = time.monotonic()
    results = kb.check_url_updates(auto_rescrape=False, check_structure=True)
    elapsed = time.monotonic() - start

    # The real contrast is "under a second" vs. the reported 1800s hang;
    # this bound is generous for a shared, concurrently-loaded machine.
    assert elapsed < 10, f"check_url_updates took {elapsed:.1f}s - it should fail fast"

    assert len(results['failed']) == 1
    failure = results['failed'][0]
    assert failure['doc_id'] == "disallowed-doc"
    assert "outside-allowlist" in failure['error'] or "Path outside allowed directories" in failure['error']
    assert results['unchanged'] == []
    assert results['changed'] == []


def test_allowed_document_is_unaffected(kb, tmp_path, monkeypatch):
    """A document whose filepath IS inside the allowlist still gets checked
    (and therefore still reaches requests.head) - the new guard must not
    reject everything indiscriminately."""
    uploads = tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    allowed_path = uploads / "page.md"
    allowed_path.write_text("content", encoding="utf-8")

    _make_url_doc(kb, tmp_path, doc_id="allowed-doc", filepath=str(allowed_path.resolve()))

    called = {"head": False}

    class _Resp:
        status_code = 200
        headers = {}

    def _fake_head(*args, **kwargs):
        called["head"] = True
        return _Resp()

    monkeypatch.setattr(requests, "head", _fake_head)

    results = kb.check_url_updates(auto_rescrape=False, check_structure=False)

    assert called["head"] is True
    assert results['failed'] == []
    assert len(results['unchanged']) == 1
    assert results['unchanged'][0]['doc_id'] == "allowed-doc"

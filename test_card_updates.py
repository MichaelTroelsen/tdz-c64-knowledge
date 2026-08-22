#!/usr/bin/env python3
"""
Acceptance tests for in-place knowledge-card updates.

Covers the six acceptance criteria from the card-update feature spec:
  1. UPSERT - refuse-by-default, replace-with-flag
  2. No orphaned chunks after a card is superseded
  3. Idempotent re-ingest via update_document
  4. Cross-card edges resolve to exactly one live card
  5. Non-card documents (no json id block) are unaffected
  6. Derived entities/relationships are refreshed, not left stale
"""
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from server import KnowledgeBase, KnowledgeBaseError, DocumentNotFoundError


def make_card_file(dirpath, filename, card_id, body, derives_from=None, extra=None):
    """Write a minimal knowledge-card markdown file with a fenced json id block."""
    data = {
        "id": card_id,
        "name": card_id,
        "edges": {
            "derives_from": derives_from or [],
            "successor_of": [],
            "shares_routine_with": []
        }
    }
    if extra:
        data.update(extra)
    content = f"# {card_id}\n\n```json\n{json.dumps(data, indent=2)}\n```\n\n{body}\n"
    path = Path(dirpath) / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def kb(temp_data_dir):
    original_allowed = os.environ.get('ALLOWED_DOCS_DIRS', '')
    original_auto_extract = os.environ.get('AUTO_EXTRACT_ENTITIES', '')
    os.environ['ALLOWED_DOCS_DIRS'] = temp_data_dir
    # Keep tests deterministic and offline: don't auto-queue background
    # (LLM-backed) entity extraction on every add_document call.
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'

    kb_instance = KnowledgeBase(temp_data_dir)
    yield kb_instance
    kb_instance.close()

    if original_allowed:
        os.environ['ALLOWED_DOCS_DIRS'] = original_allowed
    else:
        os.environ.pop('ALLOWED_DOCS_DIRS', None)
    if original_auto_extract:
        os.environ['AUTO_EXTRACT_ENTITIES'] = original_auto_extract
    else:
        os.environ.pop('AUTO_EXTRACT_ENTITIES', None)


# ---------------------------------------------------------------------------
# 1. UPSERT: refuse by default, replace with the flag
# ---------------------------------------------------------------------------

def test_upsert_refuses_then_replaces_with_flag(kb, temp_data_dir):
    f1 = make_card_file(temp_data_dir, "v1.md", "test-card",
                         "Version one body content about the wombat.")
    doc1 = kb.add_document(f1)
    assert doc1.card_id == "test-card"
    assert doc1.superseded_by is None

    f2 = make_card_file(temp_data_dir, "v2.md", "test-card",
                         "Version two body content about the aardvark.")

    # Refuses by default - silent forking is the bug this feature fixes.
    with pytest.raises(KnowledgeBaseError):
        kb.add_document(f2)

    # Explicit replace succeeds and supersedes the old one.
    doc2 = kb.add_document(f2, replace=True)
    assert doc2.doc_id != doc1.doc_id
    assert doc2.card_id == "test-card"

    live_ids = {d.doc_id for d in kb.list_documents()}
    assert doc2.doc_id in live_ids
    assert doc1.doc_id not in live_ids  # superseded, excluded from default listing

    resolved = kb.get_document_by_card_id("test-card")
    assert resolved is not None
    assert resolved.doc_id == doc2.doc_id

    old = kb.documents[doc1.doc_id]
    assert old.superseded_by == doc2.doc_id


# ---------------------------------------------------------------------------
# 2. No orphaned chunks - the regression that matters
# ---------------------------------------------------------------------------

def test_no_orphaned_chunks_after_update(kb, temp_data_dir):
    retracted_sentence = "the orderlist grammar is a fixed table of byte classes"
    f1 = make_card_file(temp_data_dir, "deenen1.md", "mon-deenen",
                         f"Some intro text. {retracted_sentence}. More text after that.")
    doc1 = kb.add_document(f1)

    f2 = make_card_file(temp_data_dir, "deenen2.md", "mon-deenen",
                         "Some intro text. The grammar is per-file, not fixed. More text after that.")
    doc2 = kb.update_document("mon-deenen", f2)
    assert doc2.doc_id != doc1.doc_id

    results = kb.search("orderlist grammar fixed table byte classes")
    assert all(r['doc_id'] != doc1.doc_id for r in results)
    assert all(retracted_sentence not in r['snippet'] for r in results)

    # And no LIVE document's content contains the retracted sentence at all.
    for doc in kb.list_documents():
        content = kb.get_document_content(doc.doc_id) or ""
        assert retracted_sentence not in content


# ---------------------------------------------------------------------------
# 3. Idempotent re-ingest via update_document
# ---------------------------------------------------------------------------

def test_idempotent_reingest(kb, temp_data_dir):
    f1 = make_card_file(temp_data_dir, "idem1.md", "idem-card",
                         "Original content for the idempotency test.")
    doc1 = kb.add_document(f1)

    f2 = make_card_file(temp_data_dir, "idem2.md", "idem-card",
                         "Updated content for the idempotency test.")
    doc2a = kb.update_document("idem-card", f2)
    doc2b = kb.update_document("idem-card", f2)  # same file, run again

    assert doc2a.doc_id == doc2b.doc_id

    live = [d for d in kb.list_documents() if d.card_id == "idem-card"]
    assert len(live) == 1
    assert live[0].doc_id == doc2a.doc_id

    # doc1 was superseded exactly once - the second update_document call was a no-op.
    old = kb.documents[doc1.doc_id]
    assert old.superseded_by == doc2a.doc_id

    # Chunks were not duplicated by the second call.
    chunks = kb._get_chunks_db(doc2a.doc_id)
    assert len(chunks) == doc2a.total_chunks
    assert len(chunks) == len({c.chunk_id for c in chunks})  # no duplicate chunk_ids


# ---------------------------------------------------------------------------
# 4. Cross-card edges resolve to exactly one live card
# ---------------------------------------------------------------------------

def test_edges_resolve_to_one_live_card(kb, temp_data_dir):
    base_v1 = make_card_file(temp_data_dir, "base1.md", "mon-deenen", "Base card body.")
    kb.add_document(base_v1)

    base_v2 = make_card_file(temp_data_dir, "base2.md", "mon-deenen", "Base card body, corrected.")
    new_base = kb.update_document("mon-deenen", base_v2)

    referencing = make_card_file(temp_data_dir, "ref.md", "some-other-card",
                                  "References mon-deenen in its edges.",
                                  derives_from=["mon-deenen"])
    kb.add_document(referencing)

    resolved = kb.get_document_by_card_id("mon-deenen")
    assert resolved is not None
    assert resolved.doc_id == new_base.doc_id

    live_matches = [d for d in kb.list_documents() if d.card_id == "mon-deenen"]
    assert len(live_matches) == 1


# ---------------------------------------------------------------------------
# 5. Non-card documents are unaffected
# ---------------------------------------------------------------------------

def test_non_card_docs_unaffected(kb, temp_data_dir):
    p1 = Path(temp_data_dir) / "doc1.txt"
    p1.write_text(
        "This is a plain text document about the VIC-II chip. No json block here.",
        encoding="utf-8"
    )
    p2 = Path(temp_data_dir) / "doc2.txt"
    p2.write_text(
        "This is a different plain text document, also with no json block, "
        "covering entirely different SID chip material.",
        encoding="utf-8"
    )

    doc1 = kb.add_document(str(p1), title="Same Title")
    doc2 = kb.add_document(str(p2), title="Same Title")

    assert doc1.doc_id != doc2.doc_id
    assert doc1.card_id is None
    assert doc2.card_id is None
    assert doc1.superseded_by is None
    assert doc2.superseded_by is None

    live_ids = {d.doc_id for d in kb.list_documents()}
    assert doc1.doc_id in live_ids
    assert doc2.doc_id in live_ids


# ---------------------------------------------------------------------------
# 6. Derived entities/relationships are refreshed, not left stale
# ---------------------------------------------------------------------------

def test_derived_entities_refresh_on_supersede(kb, temp_data_dir):
    retracted_entity = "RETRACTED_ENTITY_X"
    shared_entity = "SHARED_ENTITY_Y"

    old_body = f"Text mentioning {retracted_entity} together with {shared_entity} in one sentence."
    f1 = make_card_file(temp_data_dir, "ent1.md", "entity-card", old_body)
    doc1 = kb.add_document(f1)

    # Seed document_entities directly (bypassing the LLM) to simulate that
    # entity extraction had already run for this card.
    now = datetime.now().isoformat()
    cursor = kb.db_conn.cursor()
    cursor.execute(
        "INSERT INTO document_entities "
        "(doc_id, entity_id, entity_text, entity_type, confidence, context, "
        " occurrence_count, generated_at, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (doc1.doc_id, 1, retracted_entity, 'concept', 0.9, '', 1, now, 'test-seed')
    )
    cursor.execute(
        "INSERT INTO document_entities "
        "(doc_id, entity_id, entity_text, entity_type, confidence, context, "
        " occurrence_count, generated_at, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (doc1.doc_id, 2, shared_entity, 'concept', 0.9, '', 1, now, 'test-seed')
    )
    kb.db_conn.commit()

    # Build the co-occurrence relationship from doc1's (soon-to-be-retracted) content.
    result = kb.extract_entity_relationships(doc1.doc_id, force_regenerate=True)
    assert result['relationship_count'] >= 1
    assert len(kb.get_entity_relationships(retracted_entity)) >= 1

    # Replace doc1's content with something that mentions neither entity.
    f2 = make_card_file(temp_data_dir, "ent2.md", "entity-card",
                         "Completely different content, no tracked entities here.")
    kb.update_document("entity-card", f2)

    # The retracted entity's document_entities row for the old doc is gone.
    cursor.execute(
        "SELECT COUNT(*) FROM document_entities WHERE doc_id = ? AND entity_text = ?",
        (doc1.doc_id, retracted_entity)
    )
    assert cursor.fetchone()[0] == 0

    # No relationship involving the retracted entity survives - it no longer
    # appears in any live document's entities, and entity_relationships has
    # no doc_id column so it can only be kept correct by a full rebuild.
    assert kb.get_entity_relationships(retracted_entity) == []


# ---------------------------------------------------------------------------
# 7. remove_document purges its chunks from search, including from an
#    already-warm in-memory cache (GitHub issue #1: a deleted doc's chunks
#    stayed searchable forever in a long-running process because self.chunks
#    is only loaded from the DB once and never pruned on removal).
# ---------------------------------------------------------------------------

def test_no_orphaned_chunks_after_remove(kb, temp_data_dir):
    # Force the BM25/simple search path deterministically. The bug is
    # specifically in that path's in-memory chunk cache; the FTS5 path
    # queries the DB directly and (via a JOIN against `documents`) already
    # excludes a removed doc's chunks correctly, so it wouldn't reproduce
    # this regression, and we don't want the test's outcome to depend on
    # whatever USE_FTS5 happens to be set to in the ambient environment.
    original_fts5 = os.environ.get('USE_FTS5')
    os.environ['USE_FTS5'] = '0'
    try:
        marker = "the quokka juggles paperclips underwater on tuesdays"
        f1 = make_card_file(temp_data_dir, "gone1.md", "gone-card",
                             f"Some intro text. {marker}. More text after that.")
        doc1 = kb.add_document(f1)

        # Warm the in-memory chunk/BM25 cache *before* removal - the bug
        # only reproduces once self.chunks has already been loaded once
        # (as it would be in a long-running server after any earlier
        # search), since a search on an empty cache would just re-read the
        # (already-correct) DB.
        warm_results = kb.search(marker)
        assert any(r['doc_id'] == doc1.doc_id for r in warm_results)

        assert kb.remove_document(doc1.doc_id) is True

        results = kb.search(marker)
        assert all(r['doc_id'] != doc1.doc_id for r in results)

        # Deleted is not the same as superseded - it must not resurface even
        # when the caller explicitly asks to include superseded content.
        results_incl = kb.search(marker, include_superseded=True)
        assert all(r['doc_id'] != doc1.doc_id for r in results_incl)
    finally:
        if original_fts5 is None:
            os.environ.pop('USE_FTS5', None)
        else:
            os.environ['USE_FTS5'] = original_fts5


# ---------------------------------------------------------------------------
# 8. health_check flags documents whose recorded filepath has vanished from
#    disk. Previously only discoverable by accident - figure OCR silently
#    skips such a document ("source file is no longer on disk") and nothing
#    else reported it, so a future re-chunk/re-OCR/page-image pass would hit
#    the same wall silently.
# ---------------------------------------------------------------------------

def test_health_check_reports_missing_source_files(kb, temp_data_dir):
    f1 = make_card_file(temp_data_dir, "present.md", "present-card",
                         "This file stays on disk.")
    kb.add_document(f1)

    # use_cache=False throughout: health_check memoises for 5 minutes, so a
    # cached result from before f2 is deleted below would mask the change
    # and pass (or fail) for the wrong reason.
    clean = kb.health_check(use_cache=False)
    assert clean['metrics']['missing_source_files'] == 0
    assert clean['status'] != 'warning'

    f2 = str(Path(temp_data_dir) / "vanishing.txt")
    Path(f2).write_text("This file gets deleted after ingest.", encoding="utf-8")
    kb.add_document(f2)
    os.remove(f2)

    dirty = kb.health_check(use_cache=False)
    assert dirty['metrics']['missing_source_files'] == 1
    assert dirty['status'] == 'warning'
    assert any('filepath' in i.lower() and 'disk' in i.lower() for i in dirty['issues']), dirty['issues']


def test_health_check_missing_source_files_splits_user_vs_generated(kb, temp_data_dir):
    """missing_source_files conflates a genuinely-lost user document with a
    generated card (e.g. a DeepSID tune) whose recorded filepath was
    disposable staging never meant to persist. health_check must add
    missing_source_files_user / missing_source_files_generated beside the
    total - not redefine it - and the two must always sum back to it."""
    # A generated card: make_card_file's fenced ```json id``` block gives
    # the resulting document a card_id.
    card_path = make_card_file(temp_data_dir, "vanishing-card.md", "vanishing-card",
                                "This generated card's staging file gets deleted after ingest.")
    card_doc = kb.add_document(card_path)
    assert card_doc.card_id == "vanishing-card"
    os.remove(card_path)

    # A genuine user document: plain text, no json id block, so no card_id.
    user_path = Path(temp_data_dir) / "vanishing-user.txt"
    user_path.write_text("This is a plain user document with no card id block.", encoding="utf-8")
    user_doc = kb.add_document(str(user_path))
    assert user_doc.card_id is None
    os.remove(str(user_path))

    health = kb.health_check(use_cache=False)
    metrics = health['metrics']

    # Both halves must be non-zero, or the sum invariant below would pass
    # vacuously.
    assert metrics['missing_source_files_user'] > 0
    assert metrics['missing_source_files_generated'] > 0

    # The sum invariant this test exists to guard: the split must never
    # drift from the total that the three pre-existing tests
    # (test_card_updates.py, test_repoint_document.py, test_sid_ingest.py)
    # assert on directly.
    assert (metrics['missing_source_files_user'] + metrics['missing_source_files_generated']
            == metrics['missing_source_files'])


def test_health_check_missing_generated_card_does_not_raise_status(kb, temp_data_dir):
    """A generated card's staging file going missing is expected, not
    damage (see kb/admin.py's health_check) - health_check must escalate
    status to 'warning' on missing_source_files_user only. A deleted
    generated card should move missing_source_files_generated without
    moving status, even though the combined missing_source_files count is
    non-zero."""
    card_path = make_card_file(temp_data_dir, "vanishing-card.md", "vanishing-card",
                                "This generated card's staging file gets deleted after ingest.")
    card_doc = kb.add_document(card_path)
    assert card_doc.card_id == "vanishing-card"
    os.remove(card_path)

    health = kb.health_check(use_cache=False)
    metrics = health['metrics']

    assert metrics['missing_source_files_generated'] == 1
    assert metrics['missing_source_files_user'] == 0
    assert metrics['missing_source_files'] == 1
    assert health['status'] != 'warning'


# ---------------------------------------------------------------------------
# 9. The MCP tool count claimed in README.md, CLAUDE.md and
#    docs/ARCHITECTURE.md must equal len(TOOL_SCHEMAS). That number has been
#    hand-corrected in three separate tasks already, and re-staled hours
#    later at least twice by a commit that added a tool without touching the
#    docs - so a plain `pytest` run should catch the drift too, not just the
#    standalone validate_docs.py script run separately.
# ---------------------------------------------------------------------------

def test_documented_mcp_tool_count_matches_schemas():
    import validate_docs

    server_tool_count = len(validate_docs.extract_mcp_tools_from_server())
    claims = validate_docs.extract_claimed_tool_counts()

    for filename, found in claims.items():
        # A file with no recognisable claim fails loudly rather than being
        # skipped - see extract_claimed_tool_counts()'s docstring for why a
        # silent pass would defeat the point of this check.
        assert found, (
            f"{filename}: no MCP tool count claim found - expected wording "
            f"like 'N MCP tools' or 'N `Tool(...)` literals'"
        )
        for n in found:
            assert n == server_tool_count, (
                f"{filename}: claims {n} MCP tools, but len(TOOL_SCHEMAS) is {server_tool_count}"
            )

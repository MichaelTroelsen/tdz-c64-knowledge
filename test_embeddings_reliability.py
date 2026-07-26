#!/usr/bin/env python3
"""
Regression tests for the embeddings pipeline reliability fixes.

Background: an audit of the MCP server found that 88% of a 770-document
production knowledge base (678 documents) had silently never been embedded,
and that the mechanism for removing a document destroyed the on-disk FAISS
index for every OTHER document too. Root causes, all covered here:

  1. add_document() never loaded the embeddings model before trying to use
     it, so any session that hadn't yet run a semantic search silently
     skipped embedding every document it added - with no error or log.
  2. remove_document() nulled the in-memory index but left the on-disk
     .faiss/.json files untouched. The next add_document() then saw a "no
     index" state, built a FRESH index containing only its own new chunks,
     and overwrote the full-corpus file with it - destroying every other
     document's embeddings.
  3. There was no drift detection: a partially-populated index looks
     identical to a fully-populated one, so health_check reported no issue.
  4. There was no locking of any kind across processes, so two concurrent
     agent sessions (each Claude Code session runs its own server.py
     process) both appending to the shared embeddings files could silently
     lose one session's document entirely - reproduced directly below.

Run with:  pytest test_embeddings_reliability.py -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from server import KnowledgeBase

REPO = Path(__file__).parent
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def semantic_kb(temp_data_dir):
    """A KnowledgeBase with semantic search enabled, exactly like production."""
    original = {k: os.environ.get(k) for k in
                ('ALLOWED_DOCS_DIRS', 'AUTO_EXTRACT_ENTITIES', 'USE_SEMANTIC_SEARCH')}
    os.environ['ALLOWED_DOCS_DIRS'] = temp_data_dir
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'
    os.environ['USE_SEMANTIC_SEARCH'] = '1'

    kb_instance = KnowledgeBase(temp_data_dir)
    assert kb_instance.use_semantic, "test setup: semantic search must be enabled"
    yield kb_instance
    kb_instance.close()

    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _write_doc(dirpath, name, text):
    path = Path(dirpath) / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _embedded_doc_ids(kb):
    return {d for d, _ in kb.embeddings_doc_map}


# ---------------------------------------------------------------------------
# 1. add_document must actually embed, even on the very first call in a
#    fresh process that has never run a semantic search.
# ---------------------------------------------------------------------------

def test_add_document_embeds_without_prior_semantic_search(semantic_kb, temp_data_dir):
    """Regression: add_document used to silently skip embedding entirely.

    _add_chunks_to_embeddings() no-ops when embeddings_model is None, and
    nothing loaded the model before add_document called it - so an
    ingest-only session (the common case: nothing has searched yet) added
    documents that were permanently invisible to semantic_search, with no
    error anywhere.
    """
    assert semantic_kb.embeddings_model is None, "test setup: model must not be pre-loaded"

    f = _write_doc(temp_data_dir, "vic2.txt",
                    "The VIC-II chip handles sprites and multicolor character mode.")
    doc = semantic_kb.add_document(f)

    assert semantic_kb.embeddings_model is not None, (
        "add_document did not load the embeddings model - it will have silently "
        "skipped embedding this document"
    )
    assert doc.doc_id in _embedded_doc_ids(semantic_kb), (
        "document was added but never embedded"
    )


# ---------------------------------------------------------------------------
# 2. remove_document must not destroy other documents' embeddings.
# ---------------------------------------------------------------------------

def test_remove_document_preserves_other_documents_embeddings(semantic_kb, temp_data_dir):
    """Regression: remove_document nulled the index; the next add rebuilt it
    from scratch with only the new document, overwriting the shared file and
    silently destroying every previously-embedded document's vectors.
    """
    f1 = _write_doc(temp_data_dir, "a.txt", "SID chip audio synthesis and ADSR envelopes.")
    f2 = _write_doc(temp_data_dir, "b.txt", "CIA timer chip interrupt handling routines.")
    doc1 = semantic_kb.add_document(f1)
    doc2 = semantic_kb.add_document(f2)
    assert {doc1.doc_id, doc2.doc_id} <= _embedded_doc_ids(semantic_kb)

    semantic_kb.remove_document(doc1.doc_id)

    # doc2's vectors must survive the removal of doc1, both in memory...
    assert doc2.doc_id in _embedded_doc_ids(semantic_kb)
    assert doc1.doc_id not in _embedded_doc_ids(semantic_kb)
    # ...and on disk, re-read from scratch.
    semantic_kb._load_embeddings()
    assert doc2.doc_id in _embedded_doc_ids(semantic_kb), (
        "doc2's embeddings were lost from disk after removing doc1 - this is "
        "the corpus-destroying bug"
    )

    # And a subsequent add must APPEND to the survivors, not replace them.
    f3 = _write_doc(temp_data_dir, "c.txt", "6502 processor addressing modes reference.")
    doc3 = semantic_kb.add_document(f3)
    covered = _embedded_doc_ids(semantic_kb)
    assert covered == {doc2.doc_id, doc3.doc_id}, (
        f"expected exactly doc2+doc3 embedded, got {covered} - if doc2 is "
        "missing, the removal-then-add sequence rebuilt the index from just "
        "the new document again"
    )


def test_reconcile_chunk_cache_also_preserves_embeddings(semantic_kb, temp_data_dir):
    """reconcile_chunk_cache had the identical destructive pattern for every
    orphaned doc_id it pruned - same fix, same test shape.
    """
    f1 = _write_doc(temp_data_dir, "a.txt", "Memory map of zero page locations.")
    f2 = _write_doc(temp_data_dir, "b.txt", "KERNAL ROM routine entry points.")
    doc1 = semantic_kb.add_document(f1)
    doc2 = semantic_kb.add_document(f2)

    # Populate the in-memory chunk cache the way a prior BM25 search would,
    # so there is something for reconcile to detect as stale below.
    semantic_kb.chunks = semantic_kb._get_chunks_db()

    # Simulate DB-level divergence: doc1 removed from the DB directly,
    # bypassing remove_document, the way a future bug or manual DB edit might.
    semantic_kb._remove_document_db(doc1.doc_id)
    del semantic_kb.documents[doc1.doc_id]

    result = semantic_kb.reconcile_chunk_cache()
    assert doc1.doc_id in result['orphaned_doc_ids']

    assert doc2.doc_id in _embedded_doc_ids(semantic_kb)
    semantic_kb._load_embeddings()
    assert doc2.doc_id in _embedded_doc_ids(semantic_kb), (
        "doc2's embeddings were lost from disk after reconciling away doc1"
    )


# ---------------------------------------------------------------------------
# 3. health_check must surface a large embeddings coverage gap.
# ---------------------------------------------------------------------------

def test_health_check_flags_embeddings_coverage_gap(semantic_kb, temp_data_dir):
    """Regression: a partially-populated index looked identical to a fully
    populated one, since the rebuild trigger only fires when the index is
    COMPLETELY empty. health_check must now compare doc coverage explicitly.
    """
    # Add several documents while the model is never loaded, to reproduce the
    # "silently never embedded" state directly rather than via a mock.
    for i in range(8):
        f = _write_doc(temp_data_dir, f"doc{i}.txt", f"Reference material number {i} about the 6510 CPU.")
        semantic_kb.add_document(f)

    # Sabotage: pretend only the first one got embedded, simulating the
    # historical bug where USE_SEMANTIC_SEARCH was off during most of ingest.
    if len(semantic_kb.embeddings_doc_map) > 1:
        semantic_kb.embeddings_doc_map = semantic_kb.embeddings_doc_map[:1]
        semantic_kb._save_embeddings_locked()

    health = semantic_kb.health_check(quick_check=True, use_cache=False)
    assert 'embeddings_doc_coverage_pct' in health['features']
    assert health['features']['embeddings_doc_coverage_pct'] < 50
    assert any('coverage gap' in issue.lower() for issue in health['issues']), (
        f"expected a coverage-gap issue, got: {health['issues']}"
    )
    assert health['status'] == 'warning'


# ---------------------------------------------------------------------------
# 4. reconcile_embeddings must backfill exactly the missing documents.
# ---------------------------------------------------------------------------

def test_reconcile_embeddings_backfills_missing_docs(semantic_kb, temp_data_dir):
    doc_ids = []
    for i in range(5):
        f = _write_doc(temp_data_dir, f"doc{i}.txt", f"Disk drive protocol notes part {i}.")
        doc_ids.append(semantic_kb.add_document(f).doc_id)

    # Simulate the historical gap: only the first 2 documents got embedded.
    keep = {d for d in doc_ids[:2]}
    semantic_kb.embeddings_doc_map = [
        (d, c) for d, c in semantic_kb.embeddings_doc_map if d in keep
    ]
    semantic_kb._save_embeddings_locked()
    assert _embedded_doc_ids(semantic_kb) == keep

    result = semantic_kb.reconcile_embeddings()

    assert result['docs_covered_before'] == 2
    assert result['docs_covered_after'] == 5
    assert result['docs_still_missing'] == 0
    assert _embedded_doc_ids(semantic_kb) == set(doc_ids)


def test_reconcile_embeddings_respects_max_docs(semantic_kb, temp_data_dir):
    """max_docs lets a large gap be backfilled incrementally across calls."""
    doc_ids = []
    for i in range(4):
        f = _write_doc(temp_data_dir, f"doc{i}.txt", f"Cartridge port pinout reference {i}.")
        doc_ids.append(semantic_kb.add_document(f).doc_id)

    semantic_kb.embeddings_doc_map = []
    semantic_kb.embeddings_index = None
    semantic_kb.embeddings_file.unlink(missing_ok=True)
    semantic_kb.embeddings_map_file.unlink(missing_ok=True)

    result = semantic_kb.reconcile_embeddings(max_docs=2)
    assert result['docs_backfilled_this_call'] == 2
    assert result['docs_still_missing'] == 2


# ---------------------------------------------------------------------------
# 5. Two real concurrent processes must not lose either document's
#    embeddings. This is the core multi-agent data-loss bug: no locking of
#    any kind existed across processes, so the second writer silently
#    clobbered the first.
# ---------------------------------------------------------------------------

_WORKER = """
import os, sys, time
import server

doc_text = sys.argv[1]
filename = sys.argv[2]
barrier_dir = sys.argv[3]
nworkers = int(sys.argv[4])

# `import server` already constructs a module-level `server.kb` singleton
# (this is how the real MCP server's tool handlers all share one instance) -
# reuse it rather than constructing a second KnowledgeBase against the same
# fresh DB, which races the one-time table-migration checks against itself
# within this single process and is unrelated to the cross-process
# concurrency this test is actually about.
kb = server.kb
assert kb.use_semantic

path = os.path.join(os.environ['TDZ_DATA_DIR'], filename)
with open(path, 'w') as f:
    f.write(doc_text)

# Barrier: make sure both worker processes are up and about to add their
# document at (approximately) the same time, so their embeddings
# read-modify-write windows actually overlap - the scenario that silently
# lost data before the cross-process lock fix.
marker = os.path.join(barrier_dir, filename + '.ready')
open(marker, 'w').close()
deadline = time.time() + 120
while time.time() < deadline:
    if len([f for f in os.listdir(barrier_dir) if f.endswith('.ready')]) >= nworkers:
        break
    time.sleep(0.05)
else:
    raise SystemExit('barrier timed out')

doc = kb.add_document(path)
print(f"ADDED {doc.doc_id}")
"""


def test_concurrent_agents_do_not_lose_each_others_embeddings(temp_data_dir, tmp_path):
    """The core repro: two real processes add_document'ing at the same time
    must both end up embedded. Before the fix, this reliably lost one.
    """
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()

    env = dict(os.environ)
    env["TDZ_DATA_DIR"] = temp_data_dir
    env["ALLOWED_DOCS_DIRS"] = temp_data_dir
    env["AUTO_EXTRACT_ENTITIES"] = "0"
    env["USE_SEMANTIC_SEARCH"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_a = tmp_path / "a"
    log_b = tmp_path / "b"
    with open(f"{log_a}.out", "wb") as out_a, open(f"{log_a}.err", "wb") as err_a:
        proc_a = subprocess.Popen(
            [PYTHON, "-c", _WORKER, "Amiga Copper list programming notes.",
             "docA.txt", str(barrier_dir), "2"],
            env=env, cwd=str(REPO), stdout=out_a, stderr=err_a,
        )
    with open(f"{log_b}.out", "wb") as out_b, open(f"{log_b}.err", "wb") as err_b:
        proc_b = subprocess.Popen(
            [PYTHON, "-c", _WORKER, "BBC Micro MOS API call reference.",
             "docB.txt", str(barrier_dir), "2"],
            env=env, cwd=str(REPO), stdout=out_b, stderr=err_b,
        )

    rc_a = proc_a.wait(timeout=180)
    rc_b = proc_b.wait(timeout=180)

    out_a_text = Path(f"{log_a}.out").read_text(errors="replace")
    out_b_text = Path(f"{log_b}.out").read_text(errors="replace")
    err_a_text = Path(f"{log_a}.err").read_text(errors="replace")
    err_b_text = Path(f"{log_b}.err").read_text(errors="replace")

    assert rc_a == 0, f"worker A failed:\n{err_a_text[-3000:]}"
    assert rc_b == 0, f"worker B failed:\n{err_b_text[-3000:]}"

    doc_id_a = out_a_text.strip().split()[-1]
    doc_id_b = out_b_text.strip().split()[-1]
    assert doc_id_a != doc_id_b

    # Read the final on-disk state fresh, the way a third agent would.
    kb = KnowledgeBase(temp_data_dir)
    try:
        kb._load_embeddings()
        covered = _embedded_doc_ids(kb)
        assert doc_id_a in covered, (
            f"agent A's document was lost from the shared embeddings index. "
            f"covered={covered}"
        )
        assert doc_id_b in covered, (
            f"agent B's document was lost from the shared embeddings index "
            f"(the original bug: concurrent adds silently clobber each other). "
            f"covered={covered}"
        )
    finally:
        kb.close()

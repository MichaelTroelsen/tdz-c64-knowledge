"""Round-trip tests for the BLOBs stored in SQLite.

Two separate bugs motivated these, both from storing binary in the database
with one format and reading it back with another:

  - Cluster centroids were written with pickle.dumps but read back with
    np.frombuffer(blob, dtype=np.float32). Those do not round-trip:
    np.frombuffer raises "buffer size must be a multiple of element size" on a
    pickle stream, or silently returns garbage when the pickle length happens
    to be a multiple of 4.
  - Graph caches were written and read with pickle, which is
    code-execution-on-load for a database file that restore_backup can
    repopulate from anywhere. They are JSON (node_link_data) now.
"""
import numpy as np
import pytest

import server as server_module

nx = pytest.importorskip("networkx", reason="networkx required for graph cache")


@pytest.fixture
def kb(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOWED_DOCS_DIRS", str(tmp_path))
    monkeypatch.setenv("AUTO_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("USE_SEMANTIC_SEARCH", "0")
    instance = server_module.KnowledgeBase(str(tmp_path / "data"))
    try:
        yield instance
    finally:
        instance.close()


# --- cluster centroids -------------------------------------------------------

def test_centroid_survives_a_store_and_reload(kb):
    """The written format must be readable by np.frombuffer(float32)."""
    centroid = np.array([0.1, 0.25, -0.5, 0.75], dtype=np.float32)
    kb._store_clusters_to_db(
        [{
            'cluster_id': 'c0',
            'cluster_number': 0,
            'centroid': centroid,
            'num_documents': 3,
            'representative_docs': [],
            'top_terms': ['sprite'],
            'silhouette_score': 0.5,
        }],
        'kmeans',
    )

    blob = kb.db_conn.execute(
        "SELECT centroid_vector FROM clusters WHERE cluster_id = 'c0'"
    ).fetchone()[0]

    # This is exactly what visualize_cluster_dendrogram does; with the old
    # pickle.dumps writer it raised ValueError here.
    recovered = np.frombuffer(blob, dtype=np.float32)
    assert len(recovered) == len(centroid), (
        f"expected {len(centroid)} floats, decoded {len(recovered)} - the "
        "writer and reader disagree about the blob format"
    )
    assert np.allclose(recovered, centroid)


def test_centroid_blob_is_exactly_four_bytes_per_float(kb):
    """A pickle stream carries framing bytes; a raw buffer must not."""
    centroid = np.arange(8, dtype=np.float32)
    kb._store_clusters_to_db(
        [{
            'cluster_id': 'c1', 'cluster_number': 1, 'centroid': centroid,
            'num_documents': 1, 'representative_docs': [], 'top_terms': [],
            'silhouette_score': None,
        }],
        'kmeans',
    )
    blob = kb.db_conn.execute(
        "SELECT centroid_vector FROM clusters WHERE cluster_id = 'c1'"
    ).fetchone()[0]
    assert len(blob) == 8 * 4, f"blob is {len(blob)} bytes, expected 32"


def test_a_missing_centroid_stays_null(kb):
    kb._store_clusters_to_db(
        [{
            'cluster_id': 'c2', 'cluster_number': 2, 'centroid': None,
            'num_documents': 0, 'representative_docs': [], 'top_terms': [],
            'silhouette_score': None,
        }],
        'dbscan',
    )
    assert kb.db_conn.execute(
        "SELECT centroid_vector FROM clusters WHERE cluster_id = 'c2'"
    ).fetchone()[0] is None


# --- graph cache -------------------------------------------------------------

def test_graph_cache_round_trips_without_pickle(kb):
    G = nx.Graph()
    G.add_node('VIC-II', entity_type='hardware', occurrences=12)
    G.add_node('SID', entity_type='hardware', occurrences=9)
    G.add_edge('VIC-II', 'SID', weight=0.75)

    cache_id = kb._cache_graph(G)
    loaded = kb._load_cached_graph(cache_id)

    assert loaded is not None, "cached graph could not be reloaded"
    assert set(loaded.nodes) == {'VIC-II', 'SID'}
    assert loaded.nodes['VIC-II']['entity_type'] == 'hardware'
    assert loaded.nodes['VIC-II']['occurrences'] == 12
    assert loaded.edges['VIC-II', 'SID']['weight'] == 0.75


def test_stored_graph_is_json_not_pickle(kb):
    """Guard the security property, not just the round-trip."""
    import json

    G = nx.Graph()
    G.add_edge('a', 'b')
    cache_id = kb._cache_graph(G)

    blob, version = kb.db_conn.execute(
        "SELECT graph_data, graph_version FROM graph_cache WHERE cache_id = ?",
        (cache_id,)
    ).fetchone()

    assert version == 2, "graph_version must mark the JSON format"
    # Parseable as JSON, and free of the pickle opcode header. networkx names
    # the edge list 'edges' in current versions and 'links' in older ones -
    # accept either rather than pinning the test to one networkx release.
    parsed = json.loads(bytes(blob).decode('utf-8'))
    assert 'nodes' in parsed
    assert 'edges' in parsed or 'links' in parsed, parsed.keys()
    assert not bytes(blob).startswith(b'\x80'), "blob looks like a pickle stream"


# --- the documents.tags column -----------------------------------------------
#
# Every reader parses this column with json.loads (see _load_documents /
# _reload_documents). update_document_tags used to write ','.join(tags), which
# is not valid JSON, so one call poisoned the row and the next document reload
# - i.e. every new session - died with JSONDecodeError and loaded no documents.


@pytest.fixture
def tagged_doc(kb, tmp_path):
    path = tmp_path / "tagme.txt"
    path.write_text("VIC-II sprite notes.", encoding="utf-8")
    return kb.add_document(str(path), tags=['sid'])


def _stored_tags(kb, doc_id):
    return kb.db_conn.execute(
        "SELECT tags FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]


def test_update_document_tags_writes_json(kb, tagged_doc):
    import json

    kb.update_document_tags(tagged_doc.doc_id, ['sid', 'music'])
    raw = _stored_tags(kb, tagged_doc.doc_id)

    assert json.loads(raw) == ['sid', 'music'], (
        f"tags column is not valid JSON: {raw!r} - every reader uses json.loads"
    )


def test_documents_still_load_after_a_tag_update(kb, tagged_doc):
    """The actual failure mode: a poisoned row broke loading for everyone."""
    kb.update_document_tags(tagged_doc.doc_id, ['sid', 'music'])

    kb._reload_documents()  # what a fresh session does on startup

    assert tagged_doc.doc_id in kb.documents, "document vanished after reload"
    assert kb.documents[tagged_doc.doc_id].tags == ['sid', 'music']


def test_empty_tags_round_trip(kb, tagged_doc):
    kb.update_document_tags(tagged_doc.doc_id, [])
    kb._reload_documents()
    assert kb.documents[tagged_doc.doc_id].tags == []


def test_tags_containing_commas_survive(kb, tagged_doc):
    """A comma-joined format could not represent these at all."""
    kb.update_document_tags(tagged_doc.doc_id, ['sid, music', '6502'])
    kb._reload_documents()
    assert kb.documents[tagged_doc.doc_id].tags == ['sid, music', '6502']


def test_add_tags_to_document_keeps_the_column_valid(kb, tagged_doc):
    """The public wrapper delegates to update_document_tags."""
    import json

    kb.add_tags_to_document(tagged_doc.doc_id, ['music'])
    raw = _stored_tags(kb, tagged_doc.doc_id)
    assert set(json.loads(raw)) >= {'sid', 'music'}
    kb._reload_documents()
    assert tagged_doc.doc_id in kb.documents


def test_bulk_tag_update_leaves_memory_and_db_in_agreement(kb, tagged_doc):
    kb.update_tags_bulk(doc_ids=[tagged_doc.doc_id], add_tags=['assembly'])

    in_memory = kb.documents[tagged_doc.doc_id].tags
    kb._reload_documents()
    from_db = kb.documents[tagged_doc.doc_id].tags

    assert sorted(in_memory) == sorted(from_db), (
        f"in-memory {in_memory} disagrees with persisted {from_db}"
    )
    assert 'assembly' in from_db


def test_a_failed_tag_write_does_not_leave_stale_memory(kb, tagged_doc):
    """On a write failure the in-memory tags must not report the new value."""
    original = list(kb.documents[tagged_doc.doc_id].tags)

    # sqlite3.Connection is an immutable C type, so swap the whole connection
    # via the thread-local slot that the db_conn property reads.
    real_conn = kb.db_conn

    class _FailingConn:
        def cursor(self):
            raise RuntimeError("simulated DB failure")

        def rollback(self):
            pass

    kb._thread_local.conn = _FailingConn()
    try:
        with pytest.raises(RuntimeError):
            kb.update_document_tags(tagged_doc.doc_id, ['should-not-stick'])
    finally:
        kb._thread_local.conn = real_conn

    assert kb.documents[tagged_doc.doc_id].tags == original, (
        "in-memory tags kept a value the database never accepted"
    )


def test_a_fresh_database_gets_every_feature_table(kb):
    """Regression: schema creation and migration must agree on the table set.

    topics/document_topics/clusters/document_clusters were created only in
    _init_database_locked's existing-database branch, so a brand-new install
    silently lacked them and every topic-modelling and clustering tool failed
    with "no such table" - while a database that had been migrated worked
    fine, which is why this went unnoticed.
    """
    present = {
        row[0] for row in
        kb.db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    required = {
        'documents', 'chunks',                          # core
        'document_entities', 'entity_relationships',    # entities
        'extraction_jobs',                              # background jobs
        'topics', 'document_topics',                     # topic modelling
        'clusters', 'document_clusters',                 # clustering
        'graph_cache', 'graph_metrics', 'graph_paths',    # knowledge graph
        'document_figures',                              # figure OCR
        'mcp_call_log',
    }

    missing = required - present
    assert not missing, (
        f"a fresh database is missing {sorted(missing)} - tools depending on "
        "them fail with 'no such table' until some later migration runs"
    )


def test_a_legacy_pickle_row_is_discarded_not_loaded(kb):
    """Never unpickle a pre-migration row: that is the vulnerability itself."""
    import pickle
    from datetime import datetime

    hostile = pickle.dumps({'not': 'a graph'})
    kb.db_conn.execute(
        "INSERT INTO graph_cache (cache_id, graph_version, graph_data, "
        "node_count, edge_count, created_date) VALUES (?, 1, ?, 0, 0, ?)",
        ('legacyrow', hostile, datetime.now().isoformat()),
    )
    kb.db_conn.commit()

    assert kb._load_cached_graph('legacyrow') is None, (
        "a graph_version=1 (pickle) row must be treated as a cache miss"
    )
    left = kb.db_conn.execute(
        "SELECT COUNT(*) FROM graph_cache WHERE cache_id = 'legacyrow'"
    ).fetchone()[0]
    assert left == 0, "the legacy row should be deleted so it is not retried"

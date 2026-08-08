#!/usr/bin/env python3
"""
Cross-encoder reranker tests.

Off by default (USE_RERANKER=0) — kept in the tree because the eval harness
showed it moves individual rankings (e.g. promotes a correct-but-buried
SID-filter passage from a miss to rank 3) even though it currently loses net
recall/MRR against plain hybrid RRF on the 40-question eval set. That result
is not fully trustworthy: the eval's lexical ground truth can't distinguish
"reranker regression" from "reranker promoted a paraphrased answer the ground
truth doesn't recognise". These tests pin the parts that are unambiguously
correct regardless of that open question: reranking is a true no-op when
disabled or unavailable, and the passage fed to the cross-encoder is plain
prose from the real chunk, not the highlighted/truncated display snippet.
"""
import os
import shutil
import tempfile

import pytest

from server import KnowledgeBase


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def kb(temp_data_dir):
    saved = {k: os.environ.get(k) for k in
             ('ALLOWED_DOCS_DIRS', 'AUTO_EXTRACT_ENTITIES', 'USE_SEMANTIC_SEARCH', 'USE_RERANKER')}
    os.environ['ALLOWED_DOCS_DIRS'] = temp_data_dir
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'
    os.environ['USE_SEMANTIC_SEARCH'] = '1'
    os.environ.pop('USE_RERANKER', None)  # off unless a test opts in

    kb_instance = KnowledgeBase(temp_data_dir)
    yield kb_instance
    kb_instance.close()

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Disabled by default: rerank() must be a true no-op
# ---------------------------------------------------------------------------

def test_reranker_off_by_default(kb):
    assert kb.use_reranker is False


def test_rerank_returns_input_unchanged_when_disabled(kb):
    results = [{'doc_id': 'd1', 'chunk_id': 0, 'score': 0.9, 'snippet': 'a'},
               {'doc_id': 'd2', 'chunk_id': 0, 'score': 0.8, 'snippet': 'b'}]
    assert kb.rerank('any query', results) == results


def test_rerank_never_touches_the_model_when_disabled(kb, monkeypatch):
    """A disabled reranker must not pay the lazy-load cost at all - not even
    the check - or every first-stage search would eat the branch for nothing."""
    def fail(*a, **kw):
        raise AssertionError("_ensure_reranker_loaded called while disabled")

    monkeypatch.setattr(kb, '_ensure_reranker_loaded', fail)
    kb.rerank('query', [{'doc_id': 'd', 'chunk_id': 0, 'score': 1.0}])


def test_rerank_respects_top_k_even_when_disabled(kb):
    results = [{'doc_id': f'd{i}', 'chunk_id': 0, 'score': 1.0} for i in range(5)]
    assert len(kb.rerank('query', results, top_k=2)) == 2


def test_rerank_handles_empty_results(kb):
    assert kb.rerank('query', []) == []


def test_rerank_falls_back_when_model_unavailable(kb, monkeypatch):
    """USE_RERANKER=1 but the model failed to load (e.g. offline, package
    missing) must degrade to first-stage order, not raise."""
    kb.use_reranker = True
    kb._reranker_loaded = True
    kb.reranker_model = None
    monkeypatch.setattr(kb, '_ensure_reranker_loaded', lambda: None)

    results = [{'doc_id': 'd1', 'chunk_id': 0, 'score': 0.9}]
    assert kb.rerank('query', results) == results


def test_rerank_survives_a_predict_failure(kb, monkeypatch):
    """A cross-encoder exception (OOM, bad input) must not take the whole
    search down - fall back to the first-stage ranking instead."""
    kb.use_reranker = True
    kb._reranker_loaded = True

    class ExplodingModel:
        def predict(self, *a, **kw):
            raise RuntimeError("boom")

    kb.reranker_model = ExplodingModel()
    monkeypatch.setattr(kb, '_ensure_reranker_loaded', lambda: None)

    results = [{'doc_id': 'd1', 'chunk_id': 0, 'score': 0.9},
               {'doc_id': 'd2', 'chunk_id': 0, 'score': 0.8}]
    assert kb.rerank('query', results) == results


# ---------------------------------------------------------------------------
# Passage selection: plain prose from the real chunk, not the display snippet
# ---------------------------------------------------------------------------

def test_rerank_passage_short_chunk_returned_whole(kb, temp_data_dir):
    body = "The SID filter cutoff frequency is set by registers 21 and 22."
    path = os.path.join(temp_data_dir, 'short.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    kb.add_document(path, tags=['test'])
    doc_id = next(iter(kb.documents))

    result = {'doc_id': doc_id, 'chunk_id': 0}
    passage = kb._rerank_passage(result, {'filter', 'cutoff'}, max_chars=1400)
    assert passage.strip() == body


def test_rerank_passage_long_chunk_has_no_highlight_markers(kb, temp_data_dir):
    """The bug the reranker work was fixing mid-flight: _extract_snippet's
    default highlight=True wraps matched terms in ** markers and adds ellipses
    - display furniture that measurably confuses a model scoring the text."""
    body = ' '.join(['filter cutoff frequency detail sentence number %d.' % i
                     for i in range(400)])
    path = os.path.join(temp_data_dir, 'long.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    kb.add_document(path, tags=['test'])
    doc_id = next(iter(kb.documents))

    result = {'doc_id': doc_id, 'chunk_id': 0}
    passage = kb._rerank_passage(result, {'filter', 'cutoff'}, max_chars=200)
    assert '**' not in passage
    assert not passage.startswith('...')
    assert not passage.endswith('...')
    assert len(passage) <= 200


def test_rerank_passage_falls_back_to_snippet_for_missing_chunk(kb):
    result = {'doc_id': 'gone', 'chunk_id': 99, 'snippet': 'fallback text'}
    assert kb._rerank_passage(result, {'query'}, max_chars=1400) == 'fallback text'


def test_rerank_passage_no_query_terms_uses_hard_truncation(kb, temp_data_dir):
    body = 'x ' * 1000
    path = os.path.join(temp_data_dir, 'notitle.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    kb.add_document(path, tags=['test'])
    doc_id = next(iter(kb.documents))

    result = {'doc_id': doc_id, 'chunk_id': 0}
    passage = kb._rerank_passage(result, set(), max_chars=100)
    assert len(passage) <= 100


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def test_rerank_keeps_retrieval_score_alongside_rerank_score(kb, monkeypatch):
    """rerank_score must not overwrite 'score' - callers and the eval harness
    both read 'score' as the signal that put a candidate here in the first
    place, and losing it would hide which stage is responsible for a result."""
    kb.use_reranker = True
    kb._reranker_loaded = True

    class FixedModel:
        def predict(self, pairs, **kw):
            return [1.0] * len(pairs)

    kb.reranker_model = FixedModel()
    monkeypatch.setattr(kb, '_ensure_reranker_loaded', lambda: None)
    monkeypatch.setattr(kb, '_rerank_passage', lambda r, qt, mc: 'passage')

    results = [{'doc_id': 'd1', 'chunk_id': 0, 'score': 0.42}]
    reranked = kb.rerank('query', results)
    assert reranked[0]['retrieval_score'] == 0.42
    assert reranked[0]['score'] == 0.42
    assert reranked[0]['rerank_score'] == 1.0


def test_rerank_sorts_best_first(kb, monkeypatch):
    kb.use_reranker = True
    kb._reranker_loaded = True

    class ReverseScoreModel:
        def predict(self, pairs, **kw):
            return [0.1, 0.9, 0.5]

    kb.reranker_model = ReverseScoreModel()
    monkeypatch.setattr(kb, '_ensure_reranker_loaded', lambda: None)
    monkeypatch.setattr(kb, '_rerank_passage', lambda r, qt, mc: 'passage')

    results = [{'doc_id': 'a', 'chunk_id': 0, 'score': 0.9},
               {'doc_id': 'b', 'chunk_id': 0, 'score': 0.5},
               {'doc_id': 'c', 'chunk_id': 0, 'score': 0.7}]
    reranked = kb.rerank('query', results)
    assert [r['doc_id'] for r in reranked] == ['b', 'c', 'a']

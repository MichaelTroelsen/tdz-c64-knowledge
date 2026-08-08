#!/usr/bin/env python3
"""
Reciprocal Rank Fusion tests for hybrid_search.

The previous fusion blended max-normalised scores from two arms whose scores
share no scale (bm25() magnitudes vs 0-1 cosine similarity), so a result's
contribution depended on how strong the rest of its batch happened to be. RRF
uses ordinal position only, which cannot drift.
"""
import os
import shutil
import tempfile

import pytest

from server import KnowledgeBase


@pytest.fixture
def kb():
    temp_dir = tempfile.mkdtemp()
    saved = {k: os.environ.get(k) for k in
             ('ALLOWED_DOCS_DIRS', 'AUTO_EXTRACT_ENTITIES', 'HYBRID_FUSION', 'RRF_K')}
    os.environ['ALLOWED_DOCS_DIRS'] = temp_dir
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'
    os.environ.pop('HYBRID_FUSION', None)
    os.environ.pop('RRF_K', None)

    instance = KnowledgeBase(temp_dir)
    yield instance
    instance.close()
    shutil.rmtree(temp_dir, ignore_errors=True)

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def result(doc_id, chunk_id, score=1.0, similarity=None):
    r = {
        'doc_id': doc_id, 'chunk_id': chunk_id, 'filename': f'{doc_id}.txt',
        'title': doc_id, 'snippet': f'snippet {doc_id}', 'word_count': 100,
        'score': score,
    }
    if similarity is not None:
        r['similarity'] = similarity
    return r


def ids(results):
    return [(r['doc_id'], r['chunk_id']) for r in results]


# ---------------------------------------------------------------------------
# Core RRF behaviour
# ---------------------------------------------------------------------------

def test_appearing_in_both_arms_beats_leading_one(kb):
    """The property that makes RRF worth having: agreement across rankers
    outranks a single arm's top hit."""
    fts = [result('solo', 0), result('both', 0)]
    sem = [result('other', 0), result('both', 0)]

    fused = kb._fuse_rankings(fts, sem, semantic_weight=0.5, max_results=3)
    assert ids(fused)[0] == ('both', 0)


def test_score_magnitudes_cannot_distort_the_ranking(kb):
    """A huge bm25() magnitude on a low-ranked hit must not promote it -
    exactly the failure mode of normalise-then-blend."""
    fts = [result('first', 0, score=0.01), result('inflated', 0, score=99999.0)]
    sem = []

    fused = kb._fuse_rankings(fts, sem, semantic_weight=0.0, max_results=2)
    assert ids(fused)[0] == ('first', 0)


def test_rank_one_in_each_arm_scores_higher_than_rank_two(kb):
    fts = [result('a', 0), result('b', 0)]
    fused = kb._fuse_rankings(fts, [], semantic_weight=0.0, max_results=2)
    assert fused[0]['score'] > fused[1]['score']


def test_semantic_weight_one_ignores_the_keyword_arm(kb):
    fts = [result('keyword_only', 0)]
    sem = [result('semantic_only', 0)]

    fused = kb._fuse_rankings(fts, sem, semantic_weight=1.0, max_results=5)
    assert fused[0]['score'] > 0
    assert ids(fused)[0] == ('semantic_only', 0)
    assert dict(ids(fused)).get('keyword_only') is None or \
        [r for r in fused if r['doc_id'] == 'keyword_only'][0]['score'] == 0


def test_semantic_weight_zero_ignores_the_semantic_arm(kb):
    fts = [result('keyword_only', 0)]
    sem = [result('semantic_only', 0)]

    fused = kb._fuse_rankings(fts, sem, semantic_weight=0.0, max_results=5)
    assert ids(fused)[0] == ('keyword_only', 0)


def test_weight_is_clamped_to_range(kb):
    fts = [result('a', 0)]
    sem = [result('b', 0)]
    for weight in (-5.0, 7.0):
        fused = kb._fuse_rankings(fts, sem, semantic_weight=weight, max_results=5)
        assert all(r['score'] >= 0 for r in fused)


def test_respects_max_results(kb):
    fts = [result(f'd{i}', 0) for i in range(20)]
    fused = kb._fuse_rankings(fts, [], semantic_weight=0.0, max_results=5)
    assert len(fused) == 5


def test_deduplicates_by_doc_and_chunk(kb):
    fts = [result('same', 3)]
    sem = [result('same', 3, similarity=0.9)]

    fused = kb._fuse_rankings(fts, sem, semantic_weight=0.5, max_results=5)
    assert len(fused) == 1


def test_empty_arms_yield_no_results(kb):
    assert kb._fuse_rankings([], [], semantic_weight=0.5, max_results=5) == []


def test_one_empty_arm_still_returns_the_other(kb):
    fts = [result('a', 0), result('b', 0)]
    fused = kb._fuse_rankings(fts, [], semantic_weight=0.5, max_results=5)
    assert ids(fused) == [('a', 0), ('b', 0)]


def test_result_shape_is_preserved_for_callers(kb):
    fused = kb._fuse_rankings([result('a', 0)], [], semantic_weight=0.5, max_results=1)
    for key in ('doc_id', 'filename', 'title', 'chunk_id', 'score',
                'snippet', 'word_count', 'fts_score', 'semantic_score'):
        assert key in fused[0], f"missing {key}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_rrf_k_damps_top_rank_dominance(kb, monkeypatch):
    fts = [result('a', 0), result('b', 0)]

    monkeypatch.setenv('RRF_K', '1')
    sharp = kb._fuse_rankings(fts, [], semantic_weight=0.0, max_results=2)
    sharp_gap = sharp[0]['score'] - sharp[1]['score']

    monkeypatch.setenv('RRF_K', '1000')
    flat = kb._fuse_rankings(fts, [], semantic_weight=0.0, max_results=2)
    flat_gap = flat[0]['score'] - flat[1]['score']

    assert sharp_gap > flat_gap


def test_invalid_rrf_k_falls_back_to_default(kb, monkeypatch):
    monkeypatch.setenv('RRF_K', 'not-a-number')
    fused = kb._fuse_rankings([result('a', 0)], [], semantic_weight=0.0, max_results=1)
    assert fused[0]['score'] == pytest.approx(1.0 / 61)


def test_legacy_weighted_fusion_still_available(kb, monkeypatch):
    """Kept so a bad RRF rollout is an env var away from reverting."""
    monkeypatch.setenv('HYBRID_FUSION', 'weighted')
    fts = [result('first', 0, score=0.01), result('inflated', 0, score=99999.0)]

    fused = kb._fuse_rankings(fts, [], semantic_weight=0.0, max_results=2)
    # The legacy path ranks by normalised magnitude, so the inflated hit wins -
    # the behaviour RRF exists to avoid.
    assert ids(fused)[0] == ('inflated', 0)

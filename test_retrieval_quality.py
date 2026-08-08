#!/usr/bin/env python3
"""
Retrieval-quality regression tests.

Two defects these guard against, both of which silently degraded answers
rather than raising anything:

  1. Chunks are 1500 words but all-MiniLM-L6-v2 truncates at 256 word-pieces
     (~190 English words). Encoding a chunk whole meant ~87% of it never
     reached the model, so semantic search matched on chunk openings only.
     Chunks are now indexed as several encoder-sized windows, max-pooled back
     to one hit per chunk.

  2. _build_rag_context fed the LLM the ~300-char display snippet, so the
     generator answered from ~1.5KB of text while its token budget sat 90%
     unused. It now re-excerpts the full chunk at a per-source share of the
     budget.
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
             ('ALLOWED_DOCS_DIRS', 'AUTO_EXTRACT_ENTITIES', 'USE_SEMANTIC_SEARCH')}
    os.environ['ALLOWED_DOCS_DIRS'] = temp_data_dir
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'
    os.environ['USE_SEMANTIC_SEARCH'] = '1'

    kb_instance = KnowledgeBase(temp_data_dir)
    yield kb_instance
    kb_instance.close()

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# 1. Passage windowing
# ---------------------------------------------------------------------------

def test_short_text_stays_a_single_passage(kb):
    text = ' '.join(f'word{i}' for i in range(50))
    assert kb._embedding_passages(text) == [text]


def test_long_chunk_splits_into_encoder_sized_windows(kb):
    text = ' '.join(f'word{i}' for i in range(1500))
    passages = kb._embedding_passages(text)

    assert len(passages) > 1
    # No window may exceed the configured limit, or the encoder truncates again.
    window = int(os.getenv('EMBEDDING_WINDOW_WORDS', '200'))
    assert all(len(p.split()) <= window for p in passages)


def test_windows_cover_the_whole_chunk(kb):
    text = ' '.join(f'word{i}' for i in range(1500))
    covered = set()
    for passage in kb._embedding_passages(text):
        covered.update(passage.split())

    # The tail is the part the old whole-chunk encoding lost entirely.
    assert 'word1499' in covered
    assert len(covered) == 1500


def test_windows_overlap_so_concepts_are_not_split_across_a_boundary(kb):
    text = ' '.join(f'word{i}' for i in range(1500))
    passages = kb._embedding_passages(text)

    first_end = set(passages[0].split()[-10:])
    second_start = set(passages[1].split()[:40])
    assert first_end & second_start


def test_window_disabled_restores_whole_chunk_behaviour(kb, monkeypatch):
    monkeypatch.setenv('EMBEDDING_WINDOW_WORDS', '0')
    text = ' '.join(f'word{i}' for i in range(1500))
    assert kb._embedding_passages(text) == [text]


def test_empty_text_yields_no_passages(kb):
    assert kb._embedding_passages('   ') == []


# ---------------------------------------------------------------------------
# 2. Retrieval reaches content buried deep inside a chunk
# ---------------------------------------------------------------------------

def _write_doc(dirpath, name, body):
    path = os.path.join(dirpath, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    return path


@pytest.mark.skipif(not os.getenv('RUN_SEMANTIC_TESTS'),
                    reason="loads sentence-transformers; set RUN_SEMANTIC_TESTS=1")
def test_content_past_the_encoder_limit_is_retrievable(kb, temp_data_dir):
    """The distinguishing sentence sits at word ~900 of a 1200-word chunk -
    far past the 256 word-piece cutoff, so the old whole-chunk embedding
    could not represent it at all."""
    filler = ' '.join(['The Commodore 64 was a popular home computer.'] * 100)
    buried = ('The SID chip filter cutoff frequency is set by registers 21 and 22 '
              'at addresses 54293 and 54294.')
    body = f"{filler}\n\n{buried}\n\n{filler}"
    _write_doc(temp_data_dir, 'buried.txt', body)

    kb.add_document(os.path.join(temp_data_dir, 'buried.txt'), tags=['test'])

    results = kb.semantic_search('SID filter cutoff frequency registers', max_results=3)
    assert results, "semantic search returned nothing"
    assert any('buried.txt' in r['filename'] for r in results)


@pytest.mark.skipif(not os.getenv('RUN_SEMANTIC_TESTS'),
                    reason="loads sentence-transformers; set RUN_SEMANTIC_TESTS=1")
def test_one_result_per_chunk_despite_multiple_windows(kb, temp_data_dir):
    """A chunk occupies several index positions now; semantic_search must
    max-pool them or a single verbose chunk fills the whole result window."""
    body = ' '.join(['VIC-II raster interrupt sprite collision detection.'] * 300)
    _write_doc(temp_data_dir, 'verbose.txt', body)
    kb.add_document(os.path.join(temp_data_dir, 'verbose.txt'), tags=['test'])

    results = kb.semantic_search('raster interrupt', max_results=5)
    keys = [(r['doc_id'], r['chunk_id']) for r in results]
    assert len(keys) == len(set(keys)), f"duplicate chunks in results: {keys}"


# ---------------------------------------------------------------------------
# 3. RAG context carries real passages, not 300-char snippets
# ---------------------------------------------------------------------------

def test_rag_context_uses_full_chunk_not_display_snippet(kb, temp_data_dir):
    body = ' '.join(f'sid{i} filter cutoff' for i in range(600))
    _write_doc(temp_data_dir, 'ctx.txt', body)
    kb.add_document(os.path.join(temp_data_dir, 'ctx.txt'), tags=['test'])

    doc_id = next(iter(kb.documents))
    results = [{
        'doc_id': doc_id,
        'filename': 'ctx.txt',
        'title': 'ctx',
        'chunk_id': 0,
        'score': 0.9,
        'snippet': 'tiny snippet',
    }]

    context = kb._build_rag_context(results, 'filter cutoff')

    assert 'tiny snippet' not in context
    # Far more than the old ~300-char snippet, and inside the per-source budget.
    assert len(context) > 2000


def test_rag_context_keeps_every_source(kb, temp_data_dir):
    """The old budget check dropped later sources wholesale once the running
    total was exceeded, losing citation breadth."""
    for i in range(5):
        body = ' '.join([f'document{i} content about the 6510 processor.'] * 400)
        _write_doc(temp_data_dir, f'multi{i}.txt', body)
        kb.add_document(os.path.join(temp_data_dir, f'multi{i}.txt'), tags=['test'])

    results = []
    for i, (doc_id, meta) in enumerate(kb.documents.items()):
        results.append({
            'doc_id': doc_id,
            'filename': meta.filename,
            'title': meta.title,
            'chunk_id': 0,
            'score': 0.9 - i * 0.1,
            'snippet': 'snippet',
        })

    context = kb._build_rag_context(results, '6510 processor')

    for i in range(1, len(results) + 1):
        assert f'## Source {i}:' in context
    assert 'truncated for token limit' not in context


def test_rag_context_respects_the_token_budget(kb, temp_data_dir):
    body = ' '.join(['6510 processor pipeline detail.'] * 2000)
    _write_doc(temp_data_dir, 'big.txt', body)
    kb.add_document(os.path.join(temp_data_dir, 'big.txt'), tags=['test'])

    doc_id = next(iter(kb.documents))
    results = [{
        'doc_id': doc_id, 'filename': 'big.txt', 'title': 'big',
        'chunk_id': 0, 'score': 0.9, 'snippet': 's',
    }]

    context = kb._build_rag_context(results, '6510 processor', max_tokens=1000)
    assert len(context) // 4 <= 1000


def test_rag_context_falls_back_to_snippet_for_a_missing_chunk(kb):
    results = [{
        'doc_id': 'does-not-exist', 'filename': 'gone.txt', 'title': 'gone',
        'chunk_id': 99, 'score': 0.5, 'snippet': 'fallback snippet text',
    }]

    context = kb._build_rag_context(results, 'anything')
    assert 'fallback snippet text' in context


def test_rag_context_handles_no_results(kb):
    assert kb._build_rag_context([], 'question') == ""

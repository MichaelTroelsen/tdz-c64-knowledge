#!/usr/bin/env python3
"""
FTS5 query-construction regression tests.

The defect these guard against was silent and expensive. `_search_fts5` built
its MATCH expression from the raw query string, and a trailing '?' - every
natural-language question - is an FTS5 syntax error in operator position:

    sqlite3.OperationalError: fts5: syntax error near "?"

The handler caught it and returned [], which the caller could not distinguish
from "no matches", so every question fell through to BM25: a ~265s index build
on first use and ~4s per query after. USE_FTS5=1 was a no-op for question-shaped
input, and hybrid_search's keyword leg was really BM25.

Two invariants are covered:
  1. User prose never reaches FTS5 as syntax - '?', '*', ':', parentheses and
     bare AND/OR/NOT in the text are neutralised.
  2. A backend failure returns None, never [], so it can never again be
     misread as an empty result set.
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
             ('ALLOWED_DOCS_DIRS', 'AUTO_EXTRACT_ENTITIES',
              'USE_SEMANTIC_SEARCH', 'USE_FTS5')}
    os.environ['ALLOWED_DOCS_DIRS'] = temp_data_dir
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'
    os.environ['USE_SEMANTIC_SEARCH'] = '0'
    os.environ['USE_FTS5'] = '1'

    instance = KnowledgeBase(temp_data_dir)

    path = os.path.join(temp_data_dir, 'vic.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("The VIC-II raster register is at 53266 and holds the current "
                "raster line. A raster interrupt fires when the line matches. "
                "The SID sound chip base address is 54272.\n")
    instance.add_document(path, tags=['test'])

    yield instance
    instance.close()

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# 1. Expression construction
# ---------------------------------------------------------------------------

def test_trailing_question_mark_is_stripped(kb):
    """The exact character that broke every question."""
    expressions = kb._fts5_match_expressions('Where is the raster register?', [])
    assert expressions
    assert '?' not in expressions[0]


@pytest.mark.parametrize('query', [
    'What is the SID base address?',
    'How do sprites work (really)?',
    'Show me the VIC-II: raster register',
    'wildcard * and caret ^ and colon :',
    'unbalanced "quote here',
])
def test_operator_characters_never_reach_fts5(kb, query):
    for expression in kb._fts5_match_expressions(query, []):
        # Every token is a quoted string literal, so the only quotes present
        # are the delimiters we added.
        stripped = expression.replace(' AND ', ' ').replace(' OR ', ' ')
        for token in stripped.split(' '):
            assert token.startswith('"') and token.endswith('"'), token


def test_bare_boolean_words_in_prose_are_quoted(kb):
    """'cats and dogs' must search for the word 'and', not apply an operator."""
    expression = kb._fts5_match_expressions('sprites AND raster', [])[0]
    assert '"AND"' in expression


def test_hyphenated_words_stay_together(kb):
    expression = kb._fts5_match_expressions('VIC-II timing', [])[0]
    assert '"VIC-II"' in expression


def test_quoted_phrase_is_preserved(kb):
    expression = kb._fts5_match_expressions('"raster interrupt" timing',
                                            ['raster interrupt'])[0]
    assert '"raster interrupt"' in expression


def test_phrase_words_are_not_also_required_individually(kb):
    expression = kb._fts5_match_expressions('"raster interrupt"',
                                            ['raster interrupt'])[0]
    assert expression == '"raster interrupt"'


def test_embedded_quote_is_escaped(kb):
    expression = kb._fts5_match_expressions('say "hi', [])[0]
    # Doubling is FTS5's escape for a quote inside a string literal.
    assert expression.count('"') % 2 == 0


def test_punctuation_only_query_yields_no_expressions(kb):
    assert kb._fts5_match_expressions('???', []) == []
    assert kb._fts5_match_expressions('', []) == []


def test_and_is_tried_before_or(kb):
    expressions = kb._fts5_match_expressions('raster interrupt timing', [])
    assert len(expressions) == 2
    assert ' AND ' in expressions[0]
    assert ' OR ' in expressions[1]


# ---------------------------------------------------------------------------
# 2. End-to-end behaviour
# ---------------------------------------------------------------------------

def test_question_shaped_query_returns_hits(kb):
    results = kb._search_fts5('Where is the raster register?', set(), [], None, 5)
    assert results, "FTS5 found nothing for a question - the original bug"


def test_question_never_triggers_a_bm25_build(kb):
    """The 265s cost. If FTS5 answers, BM25 must stay unbuilt."""
    assert kb.bm25 is None
    kb.search('Where is the raster register?', max_results=5)
    assert kb.bm25 is None, "BM25 index was built despite FTS5 being available"


def test_or_fallback_recovers_when_and_is_too_strict(kb):
    """A question mixing corpus terms with absent ones still returns hits."""
    results = kb._search_fts5(
        'Does the raster register control quadrophonic holography?',
        set(), [], None, 5)
    assert results


def test_backend_failure_returns_none_not_empty(kb, monkeypatch):
    """The distinction the caller depends on."""
    monkeypatch.setattr(kb, '_fts5_match_expressions',
                        lambda query, phrases: ['NEAR('])  # real syntax error
    assert kb._search_fts5('anything', set(), [], None, 5) is None


def test_no_matches_returns_empty_list_not_none(kb):
    results = kb._search_fts5('zzzznonexistentterm', set(), [], None, 5)
    assert results == []


def test_search_survives_a_failing_fts5_backend(kb, monkeypatch):
    """None must degrade to a fallback, not crash or propagate."""
    monkeypatch.setattr(kb, '_fts5_match_expressions',
                        lambda query, phrases: ['NEAR('])
    results = kb.search('raster register', max_results=5)
    assert isinstance(results, list)

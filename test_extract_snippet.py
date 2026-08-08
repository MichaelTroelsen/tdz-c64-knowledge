#!/usr/bin/env python3
"""
_extract_snippet / _content_terms regression tests.

Both defects below were found live (real corpus content, real LLM calls) and
confirmed fixed against that same real data - see server.py's comments on
_content_terms and the density-ratio check in _extract_snippet for the full
account. Only the parts reliably reproducible as small synthetic fixtures are
pinned here; hand-constructing a minimal synthetic case for the alignment fix
itself proved unreliable (two attempts that worked in isolation failed once
run through pytest) and was dropped rather than shipped flaky.

  1. _content_terms: the density-scoring window in _extract_snippet does a
     raw, unweighted substring count with no stopword or length filtering.
     A claim's term set that includes short/common words - critically the
     bare digit "3" - matched inside unrelated numbers scattered throughout
     a document ("voice 3", "oscillator 3", "REG 3"), pulling the window
     toward a region that never mentioned the actual fact being checked.

  2. _extract_snippet's sentence-boundary alignment: after the density
     search finds the highest-scoring window (best_pos), the function walks
     back to the nearest preceding sentence boundary and re-extends by a
     fixed snippet_size. That can land the final window short of content
     that scored well near the ORIGINAL window's far edge - best_pos is only
     that window's left edge, not the position of the match itself. Found
     live: a claim quoting a SID chip's base address lost its own citation's
     supporting text this way, by 43 characters. The fix compares density
     (score per character) between the aligned window and the original
     best-scoring one, overriding only when density dropped substantially.

     Important limitation, also found live and NOT fully solved: this is a
     guard against GROSS loss, not a guarantee against every partial-term-
     loss case. If the aligned window still contains most of a multi-term
     match set but drops just one specific/rare term, the density ratio can
     stay above the threshold and the guard won't fire - a live claim citing
     a table-of-contents-style appendix listing showed exactly this pattern
     and was left unresolved rather than chased further.
"""
import os
import shutil
import tempfile

import pytest

from server import KnowledgeBase, _content_terms


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def kb(temp_data_dir):
    saved = {k: os.environ.get(k) for k in ('ALLOWED_DOCS_DIRS', 'AUTO_EXTRACT_ENTITIES')}
    os.environ['ALLOWED_DOCS_DIRS'] = temp_data_dir
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'

    kb_instance = KnowledgeBase(temp_data_dir)
    yield kb_instance
    kb_instance.close()

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# _content_terms
# ---------------------------------------------------------------------------

def test_content_terms_drops_stopwords():
    terms = _content_terms('This is the SID chip and it has three voices')
    assert 'the' not in terms
    assert 'is' not in terms
    assert 'it' not in terms
    assert 'and' not in terms
    assert 'has' not in terms


def test_content_terms_drops_bare_digits():
    """The specific defect found live: a lone "3" substring-matches inside
    "voice 3", "oscillator 3", "REG 3" throughout an unrelated region,
    inflating its density score well past the region that actually states
    "three ... voices"."""
    terms = _content_terms('the SID has 3 voices')
    assert '3' not in terms


def test_content_terms_keeps_specific_technical_tokens():
    terms = _content_terms('The base address of the SID 6581 is $D400 (54272)')
    assert 'sid' in terms
    assert '6581' in terms
    assert 'd400' in terms
    assert '54272' in terms
    assert 'address' in terms
    assert 'base' in terms


def test_content_terms_empty_for_pure_stopword_text():
    assert _content_terms('it is the of a') == set()


def test_uniform_repetitive_content_keeps_its_shorter_trimmed_window(kb):
    """The density-ratio check must NOT force every window back out to the
    full snippet_size - only when alignment actually lost matching content.
    In uniform/repetitive text a legitimately shorter, sentence-clean window
    is exactly the intended behavior (this is what _build_rag_context's
    token-budget trimming depends on)."""
    content = ' '.join(['6510 processor pipeline detail.'] * 500)
    terms = _content_terms('6510 processor')

    snippet = kb._extract_snippet(content, terms, snippet_size=2000, highlight=False)
    assert len(snippet) <= 2000
    assert '6510' in snippet


def test_extract_snippet_short_content_unaffected(kb):
    body = 'The SID filter cutoff frequency is set by registers 21 and 22.'
    snippet = kb._extract_snippet(body, {'filter', 'cutoff'}, snippet_size=300, highlight=False)
    assert snippet.strip() == body

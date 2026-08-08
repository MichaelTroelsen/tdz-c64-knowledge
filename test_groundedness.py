#!/usr/bin/env python3
"""
Answer-grounding verification tests.

Before this, `_generate_answer_with_llm` set confidence to a hardcoded
0.85/0.70 keyed on whether the literal string "Source N" appeared anywhere
in the model's output - a self-report of citing, not evidence the cited
passage actually supports the claim next to it. A wrong answer that cites
"Source 2" looked identical, in the response, to a right one.

`_verify_answer_grounding` closes that gap with a second LLM call that checks
each cited claim against its source passage. These tests mock `call_json`
directly rather than hitting a live API - same style as the rest of this
suite's LLM-touching tests - so they exercise the claim-splitting, prompt
construction, and fallback behavior deterministically.
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
             ('ALLOWED_DOCS_DIRS', 'AUTO_EXTRACT_ENTITIES', 'USE_ANSWER_VERIFICATION')}
    os.environ['ALLOWED_DOCS_DIRS'] = temp_data_dir
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'
    os.environ.pop('USE_ANSWER_VERIFICATION', None)  # on unless a test opts out

    kb_instance = KnowledgeBase(temp_data_dir)
    yield kb_instance
    kb_instance.close()

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class FakeLLMClient:
    """Stands in for llm_integration.LLMClient - only call_json is used by
    _verify_answer_grounding."""

    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.last_prompt = None

    def call_json(self, prompt, **kwargs):
        self.last_prompt = prompt
        if self._raises:
            raise self._raises
        return self._response


def _result(doc_id, chunk_id, snippet):
    return {'doc_id': doc_id, 'chunk_id': chunk_id, 'filename': f'{doc_id}.pdf',
            'title': doc_id, 'score': 0.9, 'snippet': snippet}


# ---------------------------------------------------------------------------
# Claim splitting and prompt construction
# ---------------------------------------------------------------------------

def test_no_citations_skips_the_llm_call_entirely(kb):
    """No 'Source N' anywhere in the answer means nothing to verify - must
    not spend a second LLM call confirming that."""
    client = FakeLLMClient(raises=AssertionError("call_json should not be invoked"))
    result = kb._verify_answer_grounding(
        'question', 'The SID chip has three voices.', [], [], client)

    assert result['confidence'] == 0.70
    assert result['unverified_claims'] == []
    assert result['checked_claims'] == 0
    assert client.last_prompt is None


def test_citations_present_but_out_of_range_are_not_checkable(kb):
    """'Source 5' with only 2 search results is not a resolvable citation -
    treat it the same as no citation, not a crash."""
    client = FakeLLMClient(raises=AssertionError("call_json should not be invoked"))
    search_results = [_result('d1', 0, 'x'), _result('d2', 0, 'y')]
    citations = [{'doc_id': 'd1', 'chunk_id': 0}]

    result = kb._verify_answer_grounding(
        'q', 'This claim cites Source 5, which does not exist.',
        citations, search_results, client)

    assert result['checked_claims'] == 0
    assert result['confidence'] == 0.85  # citations list was non-empty


def test_fully_supported_answer_gets_high_confidence(kb):
    search_results = [_result('sid', 0, 'The SID has three voices, each with an ADSR envelope.')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]
    client = FakeLLMClient(response={
        'verifications': [{'claim': 1, 'verdict': 'supported'}]
    })

    result = kb._verify_answer_grounding(
        'How many voices does the SID have?',
        'The SID chip has three voices (Source 1).',
        citations, search_results, client)

    assert result['confidence'] == 1.0
    assert result['unverified_claims'] == []
    assert result['checked_claims'] == 1


def test_unsupported_claim_is_flagged_and_lowers_confidence(kb):
    search_results = [_result('sid', 0, 'The SID chip is a sound synthesizer.')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]
    client = FakeLLMClient(response={
        'verifications': [
            {'claim': 1, 'verdict': 'supported'},
            {'claim': 2, 'verdict': 'not_mentioned'},
        ]
    })

    answer = 'The SID chip is a sound synthesizer (Source 1). It has exactly nine voices (Source 1).'
    result = kb._verify_answer_grounding('q', answer, citations, search_results, client)

    assert result['checked_claims'] == 2
    assert result['confidence'] == 0.5
    assert len(result['unverified_claims']) == 1
    assert 'nine voices' in result['unverified_claims'][0]


def test_contradicted_verdict_counts_as_unverified_not_supported(kb):
    search_results = [_result('sid', 0, 'The SID chip has three voices.')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]
    client = FakeLLMClient(response={
        'verifications': [{'claim': 1, 'verdict': 'contradicted'}]
    })

    result = kb._verify_answer_grounding(
        'q', 'The SID chip has five voices (Source 1).', citations, search_results, client)

    assert result['confidence'] == 0.0
    assert result['checked_claims'] == 1
    assert len(result['unverified_claims']) == 1


def test_missing_verdict_for_a_claim_defaults_to_unverified(kb):
    """If the model skips a claim in its response, treat it as unverified
    rather than crashing on a missing dict key or silently counting it as
    supported."""
    search_results = [_result('d1', 0, 'x'), _result('d2', 0, 'y')]
    citations = [{'doc_id': 'd1', 'chunk_id': 0}, {'doc_id': 'd2', 'chunk_id': 0}]
    client = FakeLLMClient(response={
        'verifications': [{'claim': 1, 'verdict': 'supported'}]  # claim 2 missing
    })

    answer = 'First fact here (Source 1). Second fact here (Source 2).'
    result = kb._verify_answer_grounding('q', answer, citations, search_results, client)

    assert result['checked_claims'] == 2
    assert result['confidence'] == 0.5
    assert len(result['unverified_claims']) == 1


# ---------------------------------------------------------------------------
# Failure fallback - a broken verifier must not take down answer_question
# ---------------------------------------------------------------------------

def test_call_json_exception_falls_back_to_heuristic(kb):
    search_results = [_result('sid', 0, 'The SID chip has three voices.')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]
    client = FakeLLMClient(raises=RuntimeError("API timeout"))

    result = kb._verify_answer_grounding(
        'q', 'The SID chip has three voices (Source 1).', citations, search_results, client)

    assert result['confidence'] == 0.85  # citations non-empty -> old heuristic
    assert result['unverified_claims'] == []
    assert result['checked_claims'] == 0


def test_malformed_json_response_falls_back_to_heuristic(kb):
    """extract_json raising ValueError inside call_json (unparseable
    response) must degrade the same way as a hard exception."""
    search_results = [_result('sid', 0, 'x')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]
    client = FakeLLMClient(raises=ValueError("LLM did not return valid JSON"))

    result = kb._verify_answer_grounding(
        'q', 'A claim (Source 1).', citations, search_results, client)

    assert result['confidence'] == 0.85
    assert result['checked_claims'] == 0


def test_response_missing_verifications_key_falls_back_gracefully(kb):
    search_results = [_result('sid', 0, 'x')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]
    client = FakeLLMClient(response={'unexpected': 'shape'})

    result = kb._verify_answer_grounding(
        'q', 'A claim (Source 1).', citations, search_results, client)

    # .get('verifications', []) degrades to zero verdicts, not a KeyError -
    # every claim ends up unverified rather than the call crashing.
    assert result['checked_claims'] == 1
    assert result['confidence'] == 0.0
    assert len(result['unverified_claims']) == 1


# ---------------------------------------------------------------------------
# Wiring: _generate_answer_with_llm and the USE_ANSWER_VERIFICATION toggle
# ---------------------------------------------------------------------------

def test_disabled_by_env_var_skips_verification(kb, monkeypatch):
    monkeypatch.setenv('USE_ANSWER_VERIFICATION', '0')

    def fail(*a, **kw):
        raise AssertionError("_verify_answer_grounding should not be called")

    monkeypatch.setattr(kb, '_verify_answer_grounding', fail)

    client = FakeLLMClient()
    client.call = lambda *a, **kw: 'The SID chip has three voices (Source 1).'
    search_results = [_result('sid', 0, 'The SID chip has three voices.')]

    result = kb._generate_answer_with_llm('q', 'context', search_results, client)

    assert result['confidence'] == 0.85
    assert result['unverified_claims'] == []
    assert result['checked_claims'] == 0


def test_generate_answer_with_llm_threads_verification_through(kb, monkeypatch):
    client = FakeLLMClient(response={
        'verifications': [{'claim': 1, 'verdict': 'not_mentioned'}]
    })
    client.call = lambda *a, **kw: 'The SID chip has nine voices (Source 1).'
    search_results = [_result('sid', 0, 'The SID chip has three voices.')]

    result = kb._generate_answer_with_llm('q', 'context', search_results, client)

    assert result['confidence'] == 0.0
    assert result['checked_claims'] == 1
    assert len(result['unverified_claims']) == 1

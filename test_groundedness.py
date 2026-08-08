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

import numpy as np
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


# ---------------------------------------------------------------------------
# Tier 2: local NLI entailment cross-encoder
#
# Verified against the actual model config before writing any of this:
# cross-encoder/nli-deberta-v3-base's id2label is
# {0: "contradiction", 1: "entailment", 2: "neutral"} (fetched from its
# HuggingFace config.json) - a wrong assumption about class order here would
# silently invert every verdict rather than raise anything, which is why
# _ensure_nli_loaded reads the mapping from the loaded model's own config
# instead of hardcoding it. Standard order is (contradiction=0, entailment=1,
# neutral=2) below, matching that verified mapping.
# ---------------------------------------------------------------------------

class FakeNLIModel:
    """Row order handed to predict() is preserved in the returned scores,
    same contract as the real CrossEncoder."""

    def __init__(self, rows):
        self._rows = rows  # list of [contradiction, entailment, neutral] score rows

    def predict(self, pairs, **kwargs):
        assert len(pairs) == len(self._rows)
        return np.array(self._rows)

    class _ExplodingPredict:
        def predict(self, *a, **kw):
            raise RuntimeError("boom")


def test_nli_off_by_default(kb):
    assert kb.use_nli_verification is False


def test_verify_claims_nli_returns_none_when_disabled(kb):
    """Disabled means _ensure_nli_loaded is a no-op and nli_model stays
    None - the dispatcher must read that as 'try the next backend', not
    crash on a None model."""
    claims = [{'text': 'x', 'sources': [1]}]
    assert kb._verify_claims_nli(claims, {(1, 1): 'passage'}) is None


def test_nli_entailment_argmax_maps_to_supported(kb):
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb._nli_label_indices = (0, 1, 2)  # contradiction, entailment, neutral
    kb.nli_model = FakeNLIModel([[0.05, 0.90, 0.05]])

    claims = [{'text': 'The SID has three voices.', 'sources': [1]}]
    verdicts = kb._verify_claims_nli(claims, {(1, 1): 'The SID chip has three voices.'})

    assert verdicts == {1: 'supported'}


def test_nli_contradiction_argmax_maps_to_contradicted(kb):
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb._nli_label_indices = (0, 1, 2)
    kb.nli_model = FakeNLIModel([[0.85, 0.10, 0.05]])

    claims = [{'text': 'The SID has nine voices.', 'sources': [1]}]
    verdicts = kb._verify_claims_nli(claims, {(1, 1): 'The SID chip has three voices.'})

    assert verdicts == {1: 'contradicted'}


def test_nli_neutral_argmax_maps_to_not_mentioned(kb):
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb._nli_label_indices = (0, 1, 2)
    kb.nli_model = FakeNLIModel([[0.10, 0.15, 0.75]])

    claims = [{'text': 'The SID was made by MOS Technology.', 'sources': [1]}]
    verdicts = kb._verify_claims_nli(claims, {(1, 1): 'The SID chip has three voices.'})

    assert verdicts == {1: 'not_mentioned'}


def test_nli_respects_a_non_standard_label_order(kb):
    """If _ensure_nli_loaded had resolved a swapped-model's labels to a
    different index order, verdicts must follow that order, not the
    standard one - this is the whole reason the indices are looked up
    per-model instead of hardcoded."""
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb._nli_label_indices = (1, 0, 2)  # (contra_idx, entail_idx, neutral_idx) swapped: idx0=entailment, idx1=contradiction
    # Row means [score for idx0, score for idx1, score for idx2]; idx1 wins, and idx1 is contra_idx here.
    kb.nli_model = FakeNLIModel([[0.05, 0.90, 0.05]])

    claims = [{'text': 'claim', 'sources': [1]}]
    verdicts = kb._verify_claims_nli(claims, {(1, 1): 'passage'})

    assert verdicts == {1: 'contradicted'}


def test_nli_multiple_claims_preserve_order(kb):
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb._nli_label_indices = (0, 1, 2)
    kb.nli_model = FakeNLIModel([
        [0.05, 0.90, 0.05],   # claim 1: supported
        [0.85, 0.10, 0.05],   # claim 2: contradicted
        [0.10, 0.10, 0.80],   # claim 3: not_mentioned
    ])

    claims = [{'text': f'claim {i}', 'sources': [1]} for i in range(1, 4)]
    passages = {(i, 1): 'passage' for i in range(1, 4)}
    verdicts = kb._verify_claims_nli(claims, passages)

    assert verdicts == {1: 'supported', 2: 'contradicted', 3: 'not_mentioned'}


def test_nli_predict_failure_returns_none_not_raises(kb):
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb._nli_label_indices = (0, 1, 2)
    kb.nli_model = FakeNLIModel._ExplodingPredict()

    claims = [{'text': 'claim', 'sources': [1]}]
    assert kb._verify_claims_nli(claims, {(1, 1): 'passage'}) is None


# ---------------------------------------------------------------------------
# _ensure_nli_loaded: label-index resolution from the model's own config
# ---------------------------------------------------------------------------

class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeHFModel:
    def __init__(self, id2label):
        self.config = _FakeConfig(id2label)


class _FakeCrossEncoder:
    def __init__(self, id2label):
        self.model = _FakeHFModel(id2label)

    def predict(self, *a, **kw):
        raise NotImplementedError


def test_ensure_nli_loaded_reads_label_order_from_model_config(kb, monkeypatch):
    """The core 'verify, don't assume' behavior: a differently-labelled
    checkpoint (via NLI_MODEL) must produce correct indices, not the
    hardcoded default."""
    kb.use_nli_verification = True
    swapped = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    monkeypatch.setattr('server.CrossEncoder', lambda *a, **kw: _FakeCrossEncoder(swapped))

    kb._ensure_nli_loaded()

    assert kb._nli_label_indices == (2, 0, 1)  # (contradiction, entailment, neutral)
    assert kb._nli_loaded is True


def test_ensure_nli_loaded_falls_back_to_standard_order_for_unrecognised_labels(kb, monkeypatch):
    kb.use_nli_verification = True
    weird = {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'}
    monkeypatch.setattr('server.CrossEncoder', lambda *a, **kw: _FakeCrossEncoder(weird))

    kb._ensure_nli_loaded()

    assert kb._nli_label_indices == (0, 1, 2)
    assert kb._nli_loaded is True


def test_ensure_nli_loaded_degrades_on_load_failure(kb, monkeypatch):
    kb.use_nli_verification = True

    def fail(*a, **kw):
        raise RuntimeError("model download failed")

    monkeypatch.setattr('server.CrossEncoder', fail)
    kb._ensure_nli_loaded()

    assert kb.use_nli_verification is False
    assert kb.nli_model is None
    assert kb._nli_loaded is False


def test_ensure_nli_loaded_is_a_noop_when_disabled(kb, monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("CrossEncoder should not be called while disabled")

    monkeypatch.setattr('server.CrossEncoder', fail)
    kb._ensure_nli_loaded()  # use_nli_verification is False by default

    assert kb._nli_loaded is False


# ---------------------------------------------------------------------------
# Dispatcher: NLI-enabled cascades to LLM, then to the heuristic
# ---------------------------------------------------------------------------

def test_dispatcher_uses_nli_and_never_touches_llm_when_nli_succeeds(kb):
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb._nli_label_indices = (0, 1, 2)
    kb.nli_model = FakeNLIModel([[0.05, 0.90, 0.05]])

    client = FakeLLMClient(raises=AssertionError("LLM should not be called when NLI succeeds"))
    search_results = [_result('sid', 0, 'The SID chip has three voices.')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]

    result = kb._verify_answer_grounding(
        'q', 'The SID chip has three voices (Source 1).', citations, search_results, client)

    assert result['confidence'] == 1.0
    assert result['checked_claims'] == 1


def test_dispatcher_falls_back_to_llm_when_nli_unavailable(kb):
    """NLI enabled but the model never loaded (offline, package missing) -
    verification must still happen, via the LLM path, not silently vanish."""
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb.nli_model = None  # simulates a failed load that already ran once

    client = FakeLLMClient(response={'verifications': [{'claim': 1, 'verdict': 'supported'}]})
    search_results = [_result('sid', 0, 'The SID chip has three voices.')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]

    result = kb._verify_answer_grounding(
        'q', 'The SID chip has three voices (Source 1).', citations, search_results, client)

    assert result['confidence'] == 1.0
    assert result['checked_claims'] == 1


def test_dispatcher_falls_back_to_heuristic_when_both_backends_fail(kb):
    kb.use_nli_verification = True
    kb._nli_loaded = True
    kb.nli_model = None

    client = FakeLLMClient(raises=RuntimeError("API down"))
    search_results = [_result('sid', 0, 'The SID chip has three voices.')]
    citations = [{'doc_id': 'sid', 'chunk_id': 0}]

    result = kb._verify_answer_grounding(
        'q', 'The SID chip has three voices (Source 1).', citations, search_results, client)

    assert result['confidence'] == 0.85  # citations non-empty -> old heuristic
    assert result['unverified_claims'] == []
    assert result['checked_claims'] == 0

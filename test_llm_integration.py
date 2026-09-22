"""Tests for the LLM provider plumbing (R14 in CODE-REVIEW.md).

Three problems this covers:
  - A fresh SDK client was constructed on every single call, discarding its
    connection pool so each request paid a new TLS handshake.
  - No request timeout: these calls run on the background extraction worker,
    so one stalled request parks that worker and stops every queued job
    behind it.
  - call_json stripped markdown fences by dropping the first and last *lines*
    of the response, which corrupted the payload whenever a model added a
    closing remark after the fence.

No network access: the SDK client is replaced with a stub.
"""
import pytest

import llm_integration
from llm_integration import LLMClient, extract_json


# --- JSON extraction ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('Here you go:\n```json\n{"a": 1}\n```', {"a": 1}),
    # The regression: trailing commentary after the fence.
    ('```json\n{"a": 1}\n```\nHope this helps!', {"a": 1}),
    ('Sure!\n```json\n{"a": 1}\n```\nLet me know.', {"a": 1}),
    ('The answer is {"a": 1} in JSON.', {"a": 1}),
    ('```JSON\n{"a": 1}\n```', {"a": 1}),
])
def test_extract_json_handles_real_world_wrappers(raw, expected):
    assert extract_json(raw) == expected


def test_extract_json_preserves_nested_structure():
    raw = '```json\n{\n  "entities": [{"entity_text": "VIC-II", "confidence": 0.9}]\n}\n```'
    assert extract_json(raw) == {
        "entities": [{"entity_text": "VIC-II", "confidence": 0.9}]
    }


def test_extract_json_wraps_a_bare_list():
    """Valid JSON, but callers index by key - give them a predictable shape."""
    assert extract_json('[1, 2]') == {'items': [1, 2]}


@pytest.mark.parametrize("raw", ['', '   ', 'I cannot help with that.', '```\nnot json\n```'])
def test_extract_json_raises_on_unparseable(raw):
    with pytest.raises(ValueError):
        extract_json(raw)


# --- client construction -----------------------------------------------------

class _StubMessages:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)

        class _Block:
            text = '{"ok": true}'

        class _Response:
            content = [_Block()]

        return _Response()


class _StubAnthropic:
    instances = []

    def __init__(self, **kwargs):
        type(self).instances.append(kwargs)
        self.kwargs = kwargs
        self.calls = []
        self.messages = _StubMessages(self)


@pytest.fixture
def anthropic_client(monkeypatch):
    _StubAnthropic.instances = []
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')

    client = LLMClient()
    monkeypatch.setattr(client.provider, '_build_client', lambda: _StubAnthropic())
    return client


def test_the_sdk_client_is_built_once_and_reused(anthropic_client):
    for _ in range(4):
        anthropic_client.call("hello")

    assert len(_StubAnthropic.instances) == 1, (
        f"built {len(_StubAnthropic.instances)} clients for 4 calls - each one "
        "discards the connection pool and pays a fresh TLS handshake"
    )


def test_a_timeout_and_retry_budget_are_configured(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')

    client = LLMClient()
    assert client.provider.timeout > 0, (
        "no request timeout: a stalled call would park the background "
        "extraction worker indefinitely"
    )
    assert client.provider.max_retries >= 1


def test_timeout_and_retries_are_passed_to_the_sdk(monkeypatch):
    _StubAnthropic.instances = []
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')
    monkeypatch.setattr(llm_integration.AnthropicProvider, '_build_client',
                        lambda self: _StubAnthropic(timeout=self.timeout,
                                                    max_retries=self.max_retries))

    client = LLMClient(timeout=12.5, max_retries=4)
    client.call("hello")

    assert _StubAnthropic.instances[0]['timeout'] == 12.5
    assert _StubAnthropic.instances[0]['max_retries'] == 4


def test_env_vars_set_the_defaults(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')
    monkeypatch.setattr(llm_integration, 'DEFAULT_TIMEOUT_S', 33.0)
    monkeypatch.setattr(llm_integration, 'DEFAULT_MAX_RETRIES', 5)

    provider = llm_integration.AnthropicProvider()
    assert provider.timeout == 33.0
    assert provider.max_retries == 5


def test_call_json_round_trips_through_the_provider(anthropic_client):
    assert anthropic_client.call_json("give me json") == {"ok": True}


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        LLMClient()


def test_get_llm_client_returns_none_when_unconfigured(monkeypatch):
    """Callers treat None as "LLM features off" rather than crashing."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')

    assert llm_integration.get_llm_client() is None


def test_unsupported_provider_is_rejected(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'not-a-provider')
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMClient()


# --- queue_entity_extraction with no LLM configured --------------------------
#
# kb.extract_entities raises ValueError('LLM not configured...') and, until
# fixed, the background worker caught that per queued job and logged one
# "Extraction job N aborted" line per document - on an install with no
# LLM_PROVIDER/API key that is every install's default state, so every
# ingested document produced an aborted job. The fix declines in
# queue_entity_extraction itself, before a job row (or queue entry) is ever
# created, so there is nothing left for the worker to abort.

from server import KnowledgeBase


@pytest.fixture
def kb_no_llm(monkeypatch, tmp_path):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('LLM_PROVIDER', raising=False)
    monkeypatch.setenv('ALLOWED_DOCS_DIRS', str(tmp_path))
    # This is the axis under test: no automatic queueing on ingest either,
    # so the test controls exactly when queue_entity_extraction runs.
    monkeypatch.setenv('AUTO_EXTRACT_ENTITIES', '0')

    kb_instance = KnowledgeBase(str(tmp_path))
    yield kb_instance, tmp_path
    kb_instance.close()


def _make_doc(kb, tmp_path, name="doc.md"):
    p = tmp_path / name
    p.write_text("# Title\n\nSome content mentioning the VIC-II chip.\n", encoding="utf-8")
    return kb.add_document(str(p))


def test_queueing_without_an_llm_declines_once_with_a_named_reason(kb_no_llm):
    kb, tmp_path = kb_no_llm
    doc = _make_doc(kb, tmp_path)

    result = kb.queue_entity_extraction(doc.doc_id)

    assert result['queued'] is False
    assert 'LLM' in result['reason']
    assert 'LLM_PROVIDER' in result['reason']


def test_queueing_without_an_llm_creates_no_job_row(kb_no_llm):
    """The old bug: a job row got created and then failed per-document.

    Nothing should land in extraction_jobs when the decline happens at
    queue time - there is no job left for the worker to pick up and abort.
    """
    kb, tmp_path = kb_no_llm
    doc = _make_doc(kb, tmp_path)

    kb.queue_entity_extraction(doc.doc_id)

    cursor = kb.db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM extraction_jobs WHERE doc_id = ?", (doc.doc_id,))
    assert cursor.fetchone()[0] == 0


def test_many_documents_without_an_llm_produce_no_aborted_jobs(kb_no_llm, caplog):
    """The reported symptom at scale: N documents, N 'aborted' log lines."""
    import logging
    kb, tmp_path = kb_no_llm
    docs = [_make_doc(kb, tmp_path, name=f"doc{i}.md") for i in range(5)]

    with caplog.at_level(logging.INFO):
        for doc in docs:
            result = kb.queue_entity_extraction(doc.doc_id)
            assert result['queued'] is False

    assert 'aborted' not in caplog.text


def test_configuring_an_llm_later_lets_the_same_call_queue_normally(kb_no_llm, monkeypatch):
    """A caller who sets an LLM up after startup must not stay declined -
    the check is per-call, not a one-time flag latched at construction."""
    kb, tmp_path = kb_no_llm
    doc = _make_doc(kb, tmp_path)

    declined = kb.queue_entity_extraction(doc.doc_id)
    assert declined['queued'] is False

    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')

    queued = kb.queue_entity_extraction(doc.doc_id)
    assert queued['queued'] is True
    assert 'job_id' in queued

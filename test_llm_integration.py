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

#!/usr/bin/env python3
"""
LLM Integration Module for TDZ C64 Knowledge Base

Supports multiple LLM providers:
- Anthropic (Claude)
- OpenAI (GPT-4, GPT-3.5)
- Local models (optional)
"""

import os
import json
import logging
import re
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Without an explicit timeout an LLM call can hang for as long as the socket
# stays open, and these calls sit on the background extraction worker - one
# stalled request would park that worker and silently stop every queued job
# behind it.
DEFAULT_TIMEOUT_S = float(os.environ.get('LLM_TIMEOUT_S', '60'))
DEFAULT_MAX_RETRIES = int(os.environ.get('LLM_MAX_RETRIES', '2'))


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def extract_json(response: str) -> Dict[str, Any]:
    """Parse a JSON object out of an LLM response.

    Models routinely wrap JSON in a markdown fence and/or add a sentence of
    commentary around it. The previous approach dropped the first and last
    *lines* of the whole response, which only worked when the fence was
    exactly the first and last line - a trailing "Hope this helps!" silently
    corrupted the payload instead of being ignored.

    Raises ValueError if no parseable JSON object is found.
    """
    if not response or not response.strip():
        raise ValueError("LLM returned an empty response")

    candidates = []

    # 1. Anything inside a fenced block (the common case).
    candidates.extend(m.group(1) for m in _FENCE_RE.finditer(response))

    # 2. The whole response, in case it is bare JSON.
    candidates.append(response.strip())

    # 3. The widest brace-delimited span, to survive surrounding prose.
    start, end = response.find('{'), response.rfind('}')
    if start != -1 and end > start:
        candidates.append(response[start:end + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        # A bare list is valid JSON but callers index by key; wrap it so they
        # get a predictable shape rather than a TypeError deep in the caller.
        if isinstance(parsed, list):
            return {'items': parsed}

    raise ValueError("LLM did not return valid JSON")


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[float] = None, max_retries: Optional[int] = None):
        self.api_key = api_key
        self.model = model
        self.timeout = DEFAULT_TIMEOUT_S if timeout is None else timeout
        self.max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        # Built once on first use, then reused. Constructing an SDK client per
        # call throws away its connection pool, so every request paid a fresh
        # TLS handshake.
        self._client = None

    def _build_client(self):
        raise NotImplementedError("Subclasses must implement _build_client()")

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def call(self, prompt: str, **kwargs) -> str:
        """Call LLM with prompt."""
        raise NotImplementedError("Subclasses must implement call()")


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[float] = None, max_retries: Optional[int] = None):
        super().__init__(api_key, model, timeout, max_retries)
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model = model or os.environ.get('LLM_MODEL', 'claude-haiku-4-5-20251001')

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

    def _build_client(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        # timeout/max_retries verified present on anthropic.Anthropic.__init__
        # (SDK 0.75.0); the SDK does the retry/backoff itself.
        return anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def call(self, prompt: str, **kwargs) -> str:
        """Call Claude API."""
        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0.3)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI (GPT) provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[float] = None, max_retries: Optional[int] = None):
        super().__init__(api_key, model, timeout, max_retries)
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model or os.environ.get('LLM_MODEL', 'gpt-3.5-turbo')

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

    def _build_client(self):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        # The openai package is not installed here, so unlike the Anthropic
        # client these kwargs could not be verified against the real
        # signature - fall back to a bare client rather than failing outright
        # if this SDK version doesn't accept them.
        try:
            return openai.OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        except TypeError as e:
            logger.warning(
                f"openai.OpenAI() rejected timeout/max_retries ({e}); "
                "falling back to SDK defaults"
            )
            return openai.OpenAI(api_key=self.api_key)

    def call(self, prompt: str, **kwargs) -> str:
        """Call OpenAI API."""
        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0.3)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Usage:
        client = LLMClient()  # Auto-detects provider from env
        response = client.call("Your prompt here")
    """

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: Optional[float] = None,
                 max_retries: Optional[int] = None):
        """
        Initialize LLM client.

        Args:
            provider: 'anthropic', 'openai', or auto-detect from env
            api_key: API key (or use environment variable)
            model: Model name (or use environment variable)
            timeout: Per-request timeout in seconds (default: LLM_TIMEOUT_S or 60)
            max_retries: SDK-level retries (default: LLM_MAX_RETRIES or 2)
        """
        self.provider_name = provider or os.environ.get('LLM_PROVIDER', 'anthropic')

        # Initialize provider
        if self.provider_name.lower() == 'anthropic':
            self.provider = AnthropicProvider(api_key, model, timeout, max_retries)
        elif self.provider_name.lower() == 'openai':
            self.provider = OpenAIProvider(api_key, model, timeout, max_retries)
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

        logger.info(f"LLM client initialized: provider={self.provider_name}, model={self.provider.model}")

    def call(self, prompt: str, **kwargs) -> str:
        """
        Call LLM with prompt.

        Args:
            prompt: Text prompt
            **kwargs: Additional provider-specific arguments

        Returns:
            LLM response text
        """
        return self.provider.call(prompt, **kwargs)

    def call_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response.

        Args:
            prompt: Text prompt (should request JSON output)
            **kwargs: Additional provider-specific arguments

        Returns:
            Parsed JSON dictionary
        """
        response = self.call(prompt, **kwargs)
        try:
            return extract_json(response)
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response}")
            raise


def get_llm_client() -> Optional[LLMClient]:
    """
    Get LLM client if configured, otherwise return None.

    Returns:
        LLMClient instance or None if not configured
    """
    try:
        return LLMClient()
    except ValueError as e:
        logger.warning(f"LLM not configured: {e}")
        return None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with environment variables
    client = get_llm_client()

    if client:
        print("Testing LLM client...")
        response = client.call("Say 'Hello, World!' in exactly 3 words.")
        print(f"Response: {response}")

        # Test JSON parsing
        json_prompt = """Return a JSON object with these fields:
        - greeting: "Hello"
        - language: "English"
        - count: 1

        Return ONLY the JSON, no other text."""

        json_response = client.call_json(json_prompt)
        print(f"JSON Response: {json_response}")
    else:
        print("LLM client not configured. Set LLM_PROVIDER and appropriate API key.")

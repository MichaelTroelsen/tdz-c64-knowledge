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
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    def call(self, prompt: str, **kwargs) -> str:
        """Call LLM with prompt."""
        raise NotImplementedError("Subclasses must implement call()")


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model = model or os.environ.get('LLM_MODEL', 'claude-3-haiku-20240307')

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

    def call(self, prompt: str, **kwargs) -> str:
        """Call Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0.3)

        try:
            response = client.messages.create(
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

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model or os.environ.get('LLM_MODEL', 'gpt-3.5-turbo')

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

    def call(self, prompt: str, **kwargs) -> str:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        client = openai.OpenAI(api_key=self.api_key)

        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0.3)

        try:
            response = client.chat.completions.create(
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
                 model: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            provider: 'anthropic', 'openai', or auto-detect from env
            api_key: API key (or use environment variable)
            model: Model name (or use environment variable)
        """
        self.provider_name = provider or os.environ.get('LLM_PROVIDER', 'anthropic')

        # Initialize provider
        if self.provider_name.lower() == 'anthropic':
            self.provider = AnthropicProvider(api_key, model)
        elif self.provider_name.lower() == 'openai':
            self.provider = OpenAIProvider(api_key, model)
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

        # Extract JSON from response (handle markdown code blocks)
        json_text = response.strip()

        # Remove markdown code blocks if present
        if json_text.startswith('```'):
            lines = json_text.split('\n')
            json_text = '\n'.join(lines[1:-1])  # Remove first and last lines

        # Remove 'json' language identifier
        if json_text.startswith('json\n'):
            json_text = json_text[5:]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response}")
            raise ValueError(f"LLM did not return valid JSON: {e}")


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

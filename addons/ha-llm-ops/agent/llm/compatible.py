"""OpenAI-compatible LLM adapter (e.g. Requesty.ai)."""

from __future__ import annotations

import os
from typing import Any

import requests

from .base import LLM

_STATUS_OK = "ok"
_STATUS_WARNING = "warning"
_STATUS_ERROR = "error"


class _RateLimitError(RuntimeError):
    """Raised when the provider signals rate limit or payment required."""


class OpenAICompatible(LLM):
    """LLM adapter for OpenAI-compatible chat completions APIs.

    Sends requests to ``{base_url}/chat/completions`` and supports an
    optional base URL (e.g. ``https://router.requesty.ai/v1``).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not set")
        self.model = model
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.timeout = timeout
        self._last_status: dict[str, Any] = {
            "provider": "compatible",
            "model": self.model,
            "level": _STATUS_OK,
            "warning": "",
        }

    def generate(self, prompt: str, *, timeout: float | None = None) -> str:
        """Return a chat-completion response for ``prompt``."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        system_prompt = "Respond with **only** valid JSON per schema below; no prose."
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        effective_timeout = timeout if timeout is not None else self.timeout
        resp = requests.post(
            url, headers=headers, json=payload, timeout=effective_timeout
        )
        if resp.status_code in (402, 429):
            self._last_status = {
                "provider": "compatible",
                "model": self.model,
                "level": _STATUS_WARNING,
                "warning": (
                    f"Provider rate-limited or payment required "
                    f"(HTTP {resp.status_code})"
                ),
            }
            raise _RateLimitError(
                f"HTTP {resp.status_code}: rate limit or payment required"
            )
        resp.raise_for_status()
        data = resp.json()

        # Extract content from OpenAI-compatible chat completions response
        choices = data.get("choices", [])
        if choices and isinstance(choices, list):
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                self._last_status = {
                    "provider": "compatible",
                    "model": self.model,
                    "level": _STATUS_OK,
                    "warning": "",
                }
                return content

        self._last_status = {
            "provider": "compatible",
            "model": self.model,
            "level": _STATUS_ERROR,
            "warning": "No text content in response",
        }
        raise RuntimeError("No text content in response")

    def get_status(self) -> dict[str, Any]:
        """Return the last known provider status."""
        return dict(self._last_status)


__all__ = ["OpenAICompatible", "_RateLimitError"]

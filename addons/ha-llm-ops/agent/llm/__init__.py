"""LLM adapter implementations and factory."""

from __future__ import annotations

import os
from typing import Any

from .base import LLM
from .compatible import OpenAICompatible, _RateLimitError
from .mock import MockLLM
from .openai import OpenAI


class _FallbackLLM:
    """Wrapper that tries the primary LLM and falls back to a backup on rate limit."""

    def __init__(self, primary: LLM, backup: LLM | None) -> None:
        self._primary = primary
        self._backup = backup

    def generate(self, prompt: str, *, timeout: float) -> str:
        """Generate with ``primary``, fall back to ``backup`` on rate limit."""
        try:
            return self._primary.generate(prompt, timeout=timeout)
        except _RateLimitError:
            if self._backup is not None:
                return self._backup.generate(prompt, timeout=timeout)
            raise

    def status(self) -> dict[str, Any]:
        """Return combined status of primary and backup providers."""
        if hasattr(self._primary, "get_status"):
            primary_status = self._primary.get_status()
        else:
            primary_status = {
                "provider": type(self._primary).__name__,
                "model": getattr(self._primary, "model", "unknown"),
                "level": "ok",
                "warning": "",
            }
        backup_status = None
        if self._backup is not None and hasattr(self._backup, "get_status"):
            backup_status = self._backup.get_status()
        return {"primary": primary_status, "backup": backup_status}


def create_llm() -> LLM:
    """Return an ``LLM`` instance based on environment.

    Reads ``LLM_PROVIDER`` (openai|requesty|compatible|mock) and related
    variables. Falls back to ``MockLLM`` when no provider is configured.
    A backup provider can be configured via ``LLM_BACKUP_*`` variables and
    is used automatically when the primary provider hits a rate limit.
    """

    provider = os.getenv("LLM_PROVIDER", "").lower()

    # Backwards-compatible default: OpenAI when its key is present
    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"

    def _build_primary() -> LLM:
        if provider == "openai":
            return OpenAI()
        if provider in ("requesty", "compatible"):
            return OpenAICompatible(
                model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
                api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL", "https://router.requesty.ai/v1"),
            )
        if provider == "mock":
            return MockLLM()
        raise RuntimeError(f"Unknown LLM provider: {provider}")

    def _build_backup() -> LLM | None:
        backup_provider = os.getenv("LLM_BACKUP_PROVIDER", "").lower()
        if not backup_provider:
            return None
        if backup_provider == "openai":
            return OpenAI()
        if backup_provider in ("requesty", "compatible"):
            return OpenAICompatible(
                model=os.getenv(
                    "LLM_BACKUP_MODEL", "nvidia/nemotron-3-super-120b-a12b"
                ),
                api_key=os.getenv("LLM_BACKUP_API_KEY"),
                base_url=os.getenv(
                    "LLM_BACKUP_BASE_URL", "https://router.requesty.ai/v1"
                ),
            )
        if backup_provider == "mock":
            return MockLLM()
        raise RuntimeError(f"Unknown backup LLM provider: {backup_provider}")

    primary = _build_primary()
    backup = _build_backup()

    if backup is not None:
        return _FallbackLLM(primary, backup)
    return primary


__all__ = ["LLM", "MockLLM", "OpenAI", "OpenAICompatible", "create_llm"]

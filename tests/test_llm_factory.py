from typing import Any

import pytest

from agent.llm import MockLLM, OpenAI, OpenAICompatible, create_llm
from agent.llm.compatible import _RateLimitError


def test_create_llm_uses_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(create_llm(), OpenAI)


def test_create_llm_defaults_to_mock(monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = create_llm()
    assert isinstance(llm, MockLLM)


def test_create_llm_requesty_provider(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "requesty")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = create_llm()
    assert isinstance(llm, OpenAICompatible)
    assert llm.model == "nemotron-3-super-120b-a12b"


def test_create_llm_compatible_provider(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://custom.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = create_llm()
    assert isinstance(llm, OpenAICompatible)
    assert llm.model == "custom-model"
    assert llm.base_url == "https://custom.example.com/v1"


def test_create_llm_mock_provider(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = create_llm()
    assert isinstance(llm, MockLLM)


def test_create_llm_unknown_provider(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "nonexistent")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Unknown LLM provider"):
        create_llm()


def test_create_llm_with_backup(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "requesty")
    monkeypatch.setenv("LLM_API_KEY", "primary-key")
    monkeypatch.setenv("LLM_BACKUP_PROVIDER", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = create_llm()
    from agent.llm import _FallbackLLM

    assert isinstance(llm, _FallbackLLM)


def test_create_llm_backup_requesty(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_BACKUP_PROVIDER", "requesty")
    monkeypatch.setenv("LLM_BACKUP_API_KEY", "backup-key")
    monkeypatch.setenv("LLM_BACKUP_MODEL", "backup-model")
    llm = create_llm()
    from agent.llm import _FallbackLLM

    assert isinstance(llm, _FallbackLLM)
    assert isinstance(llm._backup, OpenAICompatible)
    assert llm._backup.model == "backup-model"


def test_fallback_uses_backup_on_rate_limit(monkeypatch: Any) -> None:
    from agent.llm import _FallbackLLM

    class FailingPrimary:
        model = "primary"

        def get_status(self) -> dict[str, Any]:
            return {
                "provider": "primary",
                "model": "primary",
                "level": "warning",
                "warning": "rate limited",
            }

        def generate(self, prompt: str, *, timeout: float) -> str:
            raise _RateLimitError("HTTP 429")

    class WorkingBackup:
        model = "backup"

        def get_status(self) -> dict[str, Any]:
            return {
                "provider": "backup",
                "model": "backup",
                "level": "ok",
                "warning": "",
            }

        def generate(self, prompt: str, *, timeout: float) -> str:
            return "backup result"

    fallback = _FallbackLLM(FailingPrimary(), WorkingBackup())
    result = fallback.generate("hi", timeout=1)
    assert result == "backup result"

    status = fallback.status()
    assert status["primary"]["level"] == "warning"
    assert status["backup"]["level"] == "ok"


def test_fallback_raises_when_no_backup(monkeypatch: Any) -> None:
    from agent.llm import _FallbackLLM

    class FailingPrimary:
        model = "primary"

        def get_status(self) -> dict[str, Any]:
            return {
                "provider": "primary",
                "model": "primary",
                "level": "error",
                "warning": "error",
            }

        def generate(self, prompt: str, *, timeout: float) -> str:
            raise _RateLimitError("HTTP 429")

    fallback = _FallbackLLM(FailingPrimary(), None)
    with pytest.raises(_RateLimitError):
        fallback.generate("hi", timeout=1)


def test_openai_requires_api_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAI()


def test_openai_uses_env_key(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    llm = OpenAI()
    assert llm.api_key == "k"

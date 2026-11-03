from typing import Any

import pytest

from agent.llm.compatible import OpenAICompatible, _RateLimitError


class DummyResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_generate_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": '{"summary": "test"}'}}]}

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> DummyResponse:
        assert url == "https://router.requesty.ai/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "nemotron-3-super-120b-a12b"
        assert json["messages"][1]["content"] == "hi"
        return DummyResponse(payload)

    monkeypatch.setattr("agent.llm.compatible.requests.post", fake_post)
    llm = OpenAICompatible(
        model="nemotron-3-super-120b-a12b",
        api_key="test-key",
        base_url="https://router.requesty.ai/v1",
    )
    out = llm.generate("hi", timeout=1)
    assert out == '{"summary": "test"}'
    status = llm.get_status()
    assert status["level"] == "ok"
    assert status["provider"] == "compatible"
    assert status["model"] == "nemotron-3-super-120b-a12b"
    assert status["warning"] == ""


def test_generate_rate_limit_429(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> DummyResponse:
        return DummyResponse({}, status_code=429)

    monkeypatch.setattr("agent.llm.compatible.requests.post", fake_post)
    llm = OpenAICompatible(
        model="nemotron-3-super-120b-a12b",
        api_key="test-key",
        base_url="https://router.requesty.ai/v1",
    )
    with pytest.raises(_RateLimitError):
        llm.generate("hi", timeout=1)
    status = llm.get_status()
    assert status["level"] == "warning"
    assert "429" in status["warning"]


def test_generate_payment_required_402(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> DummyResponse:
        return DummyResponse({}, status_code=402)

    monkeypatch.setattr("agent.llm.compatible.requests.post", fake_post)
    llm = OpenAICompatible(
        model="nemotron-3-super-120b-a12b",
        api_key="test-key",
        base_url="https://router.requesty.ai/v1",
    )
    with pytest.raises(_RateLimitError):
        llm.generate("hi", timeout=1)
    status = llm.get_status()
    assert status["level"] == "warning"


def test_generate_no_content(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> DummyResponse:
        return DummyResponse({"choices": []})

    monkeypatch.setattr("agent.llm.compatible.requests.post", fake_post)
    llm = OpenAICompatible(
        model="nemotron-3-super-120b-a12b",
        api_key="test-key",
        base_url="https://router.requesty.ai/v1",
    )
    with pytest.raises(RuntimeError):
        llm.generate("hi", timeout=1)
    status = llm.get_status()
    assert status["level"] == "error"


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatible(model="nemotron-3-super-120b-a12b")


def test_uses_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "env-key")

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> DummyResponse:
        assert headers["Authorization"] == "Bearer env-key"
        return DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("agent.llm.compatible.requests.post", fake_post)
    llm = OpenAICompatible(model="nemotron-3-super-120b-a12b")
    assert llm.generate("hi", timeout=1) == "ok"


def test_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_url: list[str] = []

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> DummyResponse:
        captured_url.append(url)
        return DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("agent.llm.compatible.requests.post", fake_post)
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "https://router.requesty.ai/v1")
    llm = OpenAICompatible(model="test-model")
    llm.generate("hi", timeout=1)
    assert captured_url[0] == "https://router.requesty.ai/v1/chat/completions"


__all__ = ["DummyResponse"]

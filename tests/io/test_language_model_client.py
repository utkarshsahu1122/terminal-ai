from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from terminal_ai.io.language_model_client import OpenAIChatClient


class _DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_complete_returns_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request: Request, timeout: float) -> _DummyResponse:
        assert "chat/completions" in request.full_url
        return _DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "hello",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = OpenAIChatClient(model="gpt-test", api_key="key")
    text = client.complete(system_prompt="SYS", user_prompt="hi")
    assert text == "hello"


def test_complete_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request: Request, timeout: float):
        raise HTTPError(request.full_url, 400, "Bad Request", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = OpenAIChatClient(model="gpt-test", api_key="key")
    with pytest.raises(RuntimeError):
        client.complete(system_prompt="SYS", user_prompt="hi")


def test_complete_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request: Request, timeout: float):
        raise URLError("timeout")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = OpenAIChatClient(model="gpt-test", api_key="key")
    with pytest.raises(RuntimeError):
        client.complete(system_prompt="SYS", user_prompt="hi")


def test_complete_omits_auth_header_for_local_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request: Request, timeout: float) -> _DummyResponse:
        # Crucial check: Local providers like Ollama should not receive Bearer tokens
        assert "Authorization" not in request.headers
        return _DummyResponse({"choices": [{"message": {"content": "local-response"}}]})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    # Using 127.0.0.1 triggers the 'is_local' logic
    client = OpenAIChatClient(model="llama3", api_key="none", base_url="http://127.0.0.1:11434")
    text = client.complete(system_prompt="SYS", user_prompt="hi")
    assert text == "local-response"

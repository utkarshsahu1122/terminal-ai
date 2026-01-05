from __future__ import annotations

import json

import pytest

from terminal_ai.cli import command_cli


class _DummyClient:
    def __init__(self, *, model: str, api_key: str, base_url: str) -> None:  # noqa: D401 - signature parity
        self._payload = {
            "command": "ls",
            "explanation": "List files",
            "requires_confirmation": False,
            "follow_up": "",
        }

    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        return json.dumps(self._payload)


def test_main_prints_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(command_cli, "OpenAIChatClient", _DummyClient)
    exit_code = command_cli.main(["list", "files", "--no-exec"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Command: ls" in captured.out
    assert "Why: List files" in captured.out


def test_main_uses_embedded_prompt_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(command_cli, "OpenAIChatClient", _DummyClient)

    def _missing_prompt(name: str) -> str:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(command_cli, "load_prompt", _missing_prompt)

    exit_code = command_cli.main(["list", "files", "--no-exec"])
    assert exit_code == 0


def test_main_allows_no_api_key_for_local_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Standard: Local LLMs shouldn't require dummy keys
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(command_cli, "OpenAIChatClient", _DummyClient)
    
    # Passing a localhost URL should bypass the API key check
    exit_code = command_cli.main(["list", "--base-url", "http://localhost:11434", "--no-exec"])
    assert exit_code == 0


def test_main_respects_custom_shell_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(command_cli, "OpenAIChatClient", _DummyClient)
    
    # We check if the parser defaults to the env shell
    parser = command_cli._build_parser()
    args = parser.parse_args([])
    assert args.shell == "/bin/zsh"

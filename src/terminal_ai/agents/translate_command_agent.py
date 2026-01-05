"""Agent responsible for translating natural language into shell commands."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terminal_ai.io.language_model_client import LanguageModelClient

# Enhanced patterns for security compliance
_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"rm\s+-r[f|v|i]*\b"),
    re.compile(r"mkfs\b"),
    re.compile(r"dd\s+if=", re.IGNORECASE),
    re.compile(r"shutdown\b|reboot\b|poweroff\b"),
    re.compile(r"\|\s*(bash|sh|zsh|python|curl)\b"),  # Remote script pipes
    re.compile(r"sudo\s+"),  # Unauthorized privilege escalation
    re.compile(r"chmod\s+777"),  # Insecure permissions
)


@dataclass(slots=True)
class CommandRequest:
    """Represents a natural language instruction to convert to a shell command."""

    instruction: str
    cwd: Path | None = None
    shell: str = "/bin/bash"
    temperature: float = 0.0
    allow_destructive: bool = False


@dataclass(slots=True)
class CommandSuggestion:
    """Structured suggestion returned by the language model."""

    command: str
    explanation: str
    requires_confirmation: bool
    follow_up: str

    def with_confirmation(self, required: bool) -> "CommandSuggestion":
        return CommandSuggestion(
            command=self.command,
            explanation=self.explanation,
            requires_confirmation=required,
            follow_up=self.follow_up,
        )


class CommandParsingError(RuntimeError):
    """Raised when the model response cannot be parsed into a suggestion."""


class TranslateCommandAgent:
    """High-level agent orchestrating prompt construction and parsing."""

    def __init__(
        self,
        model_client: LanguageModelClient,
        *,
        system_prompt_template: str,
    ) -> None:
        self._model_client = model_client
        self._system_prompt_template = system_prompt_template

    def suggest(self, request: CommandRequest) -> CommandSuggestion:
        # Structure the prompt with role separation to prevent instruction injection
        system_prompt = self._system_prompt_template.format(
            shell=request.shell,
            cwd=(str(request.cwd) if request.cwd else "~"),
        )
        
        # Complete the request
        raw_response = self._model_client.complete(
            system_prompt=system_prompt,
            user_prompt=f"PROCESS DATA: {request.instruction.strip()}",
            temperature=request.temperature,
        )

        # Robust parse and enforce confirmation
        suggestion = self._parse_response(raw_response)
        if suggestion.command and not request.allow_destructive:
            suggestion = self._enforce_confirmation(suggestion)
        return suggestion

    @staticmethod
    def _parse_response(response_text: str) -> CommandSuggestion:
        """Robustly extracts JSON from potentially markdown-wrapped text."""
        try:
            payload = _extract_json_from_markdown(response_text)
        except ValueError as exc:  # pragma: no cover - defensive path
            raise CommandParsingError(f"Failed to extract JSON: {exc}") from exc

        command = str(payload.get("command", "")).strip()
        explanation = str(payload.get("explanation", "")).strip()
        follow_up = str(payload.get("follow_up", "")).strip()
        requires_confirmation = bool(payload.get("requires_confirmation", False))

        if not command and not follow_up:
            raise CommandParsingError("Model returned empty command and follow_up")

        return CommandSuggestion(
            command=command,
            explanation=explanation,
            requires_confirmation=requires_confirmation,
            follow_up=follow_up,
        )

    @staticmethod
    def _enforce_confirmation(suggestion: CommandSuggestion) -> CommandSuggestion:
        """Normalizes and scans commands against security patterns."""
        normalized_cmd = suggestion.command.lower().strip()
        if any(pattern.search(normalized_cmd) for pattern in _DESTRUCTIVE_PATTERNS):
            return suggestion.with_confirmation(True)
        return suggestion


def _extract_json_from_markdown(response_text: str) -> dict[str, object]:
    """Return the first JSON object found in the response text."""

    # Attempt to find JSON inside triple backticks first
    markdown_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if markdown_json:
        candidate = markdown_json.group(1)
    else:
        # Fallback to finding the first/last curly brace
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format: {exc}") from exc


__all__ = [
    "CommandRequest",
    "CommandSuggestion",
    "CommandParsingError",
    "TranslateCommandAgent",
]

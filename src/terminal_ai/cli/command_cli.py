"""CLI entrypoint converting natural language to shell commands."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Sequence

from terminal_ai.agents.translate_command_agent import (
    CommandParsingError,
    CommandRequest,
    TranslateCommandAgent,
)
from terminal_ai.io.command_runner import CommandRunner
from terminal_ai.io.language_model_client import OpenAIChatClient
from terminal_ai.io.prompt_loader import load_prompt

_DEFAULT_PROMPT = "command_synthesis.txt"
_DEFAULT_MODEL = os.getenv("TERMINAL_AI_MODEL", "gpt-4o-mini")
_BASE_URL = os.getenv("TERMINAL_AI_BASE_URL", "api.openai.com")

# ANSI Color codes for better UX
CLR_G = "\033[92m"  # Green
CLR_Y = "\033[93m"  # Yellow
CLR_D = "\033[2m"   # Dim
CLR_R = "\033[0m"   # Reset

_EMBEDDED_PROMPT = dedent(
    """\
    You are TerminalAI, an expert macOS and Linux shell assistant. Convert the provided
    user request into a single shell command that can be executed from an interactive
    terminal session.

    Constraints:
    - Assume the default shell is {shell} unless specified otherwise.
    - Assume the current working directory is {cwd}.
    - Prefer concise commands that rely on standard tooling already available.
    - If the request is ambiguous, add a clarifying question in the `follow_up` field.
    - Respect safety: never include commands that permanently delete data, reformat
      disks, or escalate privileges unless explicitly requested.
    - Output MUST be valid JSON matching this schema:
      {{
        "command": "...",            # string; leave empty when no safe command exists
        "explanation": "...",       # short justification for the command
        "requires_confirmation": bool,
        "follow_up": "..."          # optional; empty string when nothing to clarify
      }}
    - Keep the explanation under 160 characters.

    Render the JSON directly with no surrounding markdown. Fill in all fields.
    """
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    instruction = " ".join(args.instruction).strip()
    if not instruction:
        try:
            instruction = input(f"{CLR_G}Describe the task>{CLR_R} ").strip()
        except EOFError:
            instruction = ""
    if not instruction:
        print(f"{CLR_Y}No instruction provided.{CLR_R}", file=sys.stderr)
        return 1

    # API Key Logic: Optional for local endpoints
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    is_local = "localhost" in args.base_url or "127.0.0.1" in args.base_url
    if not api_key and not is_local:
        print("Missing API key. Set OPENAI_API_KEY or use a local provider.", file=sys.stderr)
        return 1

    # Load Prompt with Fallback
    try:
        prompt_template = load_prompt(args.prompt)
    except FileNotFoundError:
        if args.prompt != _DEFAULT_PROMPT: raise
        prompt_template = _EMBEDDED_PROMPT

    client = OpenAIChatClient(model=args.model, api_key=api_key or "local", base_url=args.base_url)
    agent = TranslateCommandAgent(model_client=client, system_prompt_template=prompt_template)

    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
    request = CommandRequest(
        instruction=instruction,
        cwd=cwd,
        shell=args.shell,
        temperature=args.temperature,
        allow_destructive=args.allow_destructive,
    )

    try:
        suggestion = agent.suggest(request)
    except CommandParsingError as exc:
        print(f"Failed to parse model response: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive guard for HTTP errors
        print(f"Model request failed: {exc}", file=sys.stderr)
        return 3

    # Handle Follow-ups
    if suggestion.follow_up and not suggestion.command:
        print(f"{CLR_Y}Follow-up needed:{CLR_R} {suggestion.follow_up}")
        return 0

    print(f"\n{CLR_G}Command:{CLR_R} {suggestion.command}")
    if suggestion.explanation:
        print(f"{CLR_D}Why: {suggestion.explanation}{CLR_R}\n")

    if args.no_exec:
        return 0

    # Confirm Execution
    should_execute = args.accept or not suggestion.requires_confirmation
    if not should_execute:
        try:
            prompt_text = f"Execute this command on {args.shell}? [y/N]: "
            answer = input(prompt_text).strip().lower()
            should_execute = answer in {"y", "yes"}
        except EOFError:
            should_execute = False

    if not should_execute:
        print("Aborted.")
        return 0

    # Execute
    runner = CommandRunner(shell=args.shell, dry_run=args.dry_run)
    result = runner.execute(suggestion.command, cwd=cwd)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(f"{CLR_Y}{result.stderr}{CLR_R}", end="", file=sys.stderr)
    return result.returncode

    if result.returncode != 0:
        print(f"\n{CLR_Y}Command failed with exit code {result.returncode}{CLR_R}")
        
    return result.returncode

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terminal AI: Natural Language to Shell")
    parser.add_argument("instruction", nargs="*", help="Natural language description of the task")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT, help="Prompt template filename inside prompts/")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Model name for the OpenAI Chat API")
    parser.add_argument("--api-key", dest="api_key", help="Override OPENAI_API_KEY environment variable")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="Override OpenAI API base URL")
    parser.add_argument("--shell", default="/bin/bash", help="Shell executable used for command execution")
    parser.add_argument("--cwd", help="Working directory for executing commands")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model")
    parser.add_argument("--dry-run", action="store_true", help="Print the command but do not execute it")
    parser.add_argument(
        "--accept",
        action="store_true",
        help="Skip the confirmation prompt and execute immediately",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="Do not execute the generated command",
    )
    parser.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Allow potentially destructive commands without forcing confirmation",
    )
    return parser


__all__ = ["main"]

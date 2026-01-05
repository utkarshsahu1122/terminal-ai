"""Utilities for executing shell commands safely."""

from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

@dataclass(slots=True)
class CommandExecutionResult:
    """Represents the outcome of a shell command execution."""

    command: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class CommandRunner:
    """Simple wrapper around ``subprocess.run`` with sensible defaults."""

    def __init__(self, shell: str = "/bin/bash", dry_run: bool = False) -> None:
        self.shell = shell
        self.dry_run = dry_run

    def execute(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandExecutionResult:
        if self.dry_run:
            print(f"[Dry Run] Would execute: {command}")
            return CommandExecutionResult(command=command, returncode=0, stdout="", stderr="")

        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        # Use Popen to allow real-time terminal output while capturing strings
        process = subprocess.Popen(
            command,
            shell=True,
            executable=self.shell,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, **(env or {})}
        )

        stdout_acc, stderr_acc = [], []
        
        # Real-time output handling
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                stdout_acc.append(line)

        _, stderr = process.communicate()
        if stderr:
            sys.stderr.write(stderr)
            stderr_acc.append(stderr)

        return CommandExecutionResult(
            command=command,
            returncode=process.returncode,
            stdout="".join(stdout_acc),
            stderr="".join(stderr_acc)
        )

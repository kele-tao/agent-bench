"""Exception hierarchy and stable error codes.

Every user-facing failure carries a machine readable code so the CLI can print
one actionable line instead of a traceback, and so results files stay greppable.
"""

from __future__ import annotations


class AgentBenchError(Exception):
    """Base class for errors the CLI knows how to render."""

    code = "agent_bench_error"
    exit_code = 2

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def render(self) -> str:
        line = f"error [{self.code}]: {self.message}"
        if self.hint:
            line += f"\n  hint: {self.hint}"
        return line


class UsageError(AgentBenchError):
    """Bad flags or a missing argument."""

    code = "usage"
    exit_code = 2


class SuiteError(AgentBenchError):
    """A suite file is missing, unreadable or fails validation."""

    code = "suite"
    exit_code = 2


class ToolInputError(AgentBenchError):
    """A tool was called with arguments that violate its schema."""

    code = "invalid_arguments"
    exit_code = 2


class AdapterError(AgentBenchError):
    """An adapter could not be constructed or failed at runtime."""

    code = "adapter"
    exit_code = 2

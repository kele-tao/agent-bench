"""Plain data structures shared by the loader, the runner, the metrics and the reports.

Nothing in this module has side effects; it exists so every other module can
import the same shapes without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CATEGORIES = (
    "tool-selection",
    "multi-step-planning",
    "error-recovery",
    "ambiguity-handling",
    "constraint-following",
    "long-context-retrieval",
    "arithmetic-with-tools",
    "refusal-safety",
)


@dataclass
class Grader:
    """One deterministic, machine-checkable expectation attached to a task."""

    kind: str
    params: dict[str, Any]
    weight: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "params": self.params,
            "weight": self.weight,
            "description": self.description,
        }


@dataclass
class Turn:
    """A message the harness feeds to the adapter."""

    role: str
    content: str
    name: str | None = None

    def render(self) -> str:
        if self.name:
            return f"{self.role} ({self.name}): {self.content}"
        return f"{self.role}: {self.content}"


@dataclass
class FaultSpec:
    tool: str
    kind: str
    times: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "kind": self.kind, "times": self.times}


@dataclass
class Task:
    """One benchmark scenario."""

    id: str
    category: str
    title: str
    instruction: str
    weight: float = 1.0
    max_steps: int = 10
    seed: int = 0
    intent: dict[str, Any] = field(default_factory=dict)
    world: dict[str, Any] = field(default_factory=dict)
    faults: list[FaultSpec] = field(default_factory=list)
    graders: list[Grader] = field(default_factory=list)
    notes: str = ""
    suite: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "instruction": self.instruction,
            "weight": self.weight,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "intent": self.intent,
            "world": self.world,
            "faults": [fault.to_dict() for fault in self.faults],
            "graders": [grader.to_dict() for grader in self.graders],
            "notes": self.notes,
        }


@dataclass
class CallRecord:
    """A single tool call as observed by the harness."""

    index: int
    tool: str
    args: dict[str, Any]
    ok: bool
    output: str | None
    error: str | None
    error_kind: str | None
    fault_injected: bool
    ticks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "error_kind": self.error_kind,
            "fault_injected": self.fault_injected,
            "ticks": self.ticks,
        }


@dataclass
class CheckOutcome:
    kind: str
    label: str
    passed: bool
    weight: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "passed": self.passed,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass
class EpisodeRecord:
    """Everything the harness observed for one task under one policy."""

    task_id: str
    category: str
    repeat: int
    policy: str
    answer: str
    stop_reason: str
    score: float
    passed: bool
    steps: int
    calls: list[CallRecord] = field(default_factory=list)
    checks: list[CheckOutcome] = field(default_factory=list)
    recovered_faults: int = 0
    injected_faults: int = 0
    claimed_tools: list[str] = field(default_factory=list)

    @property
    def failed_calls(self) -> int:
        return sum(1 for call in self.calls if not call.ok)

    @property
    def schema_violations(self) -> int:
        return sum(
            1 for call in self.calls if call.error_kind == "schema_violation" and not call.ok
        )

    @property
    def wasted_calls(self) -> int:
        return sum(1 for call in self.calls if not call.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "repeat": self.repeat,
            "policy": self.policy,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "score": round(self.score, 6),
            "passed": self.passed,
            "steps": self.steps,
            "calls": [call.to_dict() for call in self.calls],
            "checks": [check.to_dict() for check in self.checks],
            "recovered_faults": self.recovered_faults,
            "injected_faults": self.injected_faults,
            "claimed_tools": self.claimed_tools,
        }

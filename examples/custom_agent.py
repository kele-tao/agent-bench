#!/usr/bin/env python
"""Wrap your own agent in an adapter in about twenty lines.

    python examples/custom_agent.py

The adapter protocol is two methods:

    reset(task)                  -> prepare state for one episode
    next_action(observation)     -> {"type": "tool", ...} or {"type": "finish", ...}

Everything else (schema validation, fault injection, grading, metrics) is done
by the harness, so a policy cannot cheat by editing the world.

This example is a hand-written "look at the instruction, then act" policy: it
answers the health-check and notes tasks without a model, and finishes honestly
with \"unknown\" for everything else. Against the bundled suites it scores far
below the reference policy - which is the point of publishing an adapter API.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agent_bench.adapters import Observation  # noqa: E402
from agent_bench.models import Task  # noqa: E402
from agent_bench.runner import run_episode  # noqa: E402
from agent_bench.suite import load_bundled_suite  # noqa: E402


class RuleBasedAgent:
    """A tiny policy that only knows two shapes of task."""

    name = "rules"

    def __init__(self) -> None:
        self.task: Task | None = None
        self.answered = False

    def reset(self, task: Task) -> None:
        self.task = task
        self.answered = False

    def next_action(self, observation: Observation) -> dict:
        assert self.task is not None
        instruction = self.task.instruction.lower()
        if observation.last is not None and observation.last.ok:
            payload = observation.last.payload() or {}
            body = payload.get("body") or payload
            if isinstance(body, dict) and "state" in body:
                return {
                    "type": "finish",
                    "answer": f"the service is {body['state']}",
                    "claimed_tools": ["http_get"],
                }
            if isinstance(body, dict) and "content" in body:
                return {
                    "type": "finish",
                    "answer": body["content"],
                    "claimed_tools": ["fs_read"],
                }
        if self.answered:
            return {"type": "finish", "answer": "unknown", "claimed_tools": []}
        if "status api" in instruction or "/status/" in instruction:
            self.answered = True
            return {"type": "tool", "tool": "http_get", "args": {"url": "/status/payments"}}
        if "notes.txt" in instruction:
            self.answered = True
            return {"type": "tool", "tool": "fs_read", "args": {"path": "/work/notes.txt"}}
        self.answered = True
        return {"type": "finish", "answer": "I do not know how to do that.", "claimed_tools": []}


def main() -> int:
    agent = RuleBasedAgent()
    _, tasks = load_bundled_suite("core")
    passed = 0
    for task in tasks:
        record = run_episode(task, agent)
        passed += int(record.passed)
        mark = "PASS" if record.passed else "FAIL"
        print(f"{mark}  {task.id:<28} {record.score * 100:5.1f}%  {record.stop_reason}")
    print()
    print(f"rules agent: {passed}/{len(tasks)} tasks, "
          f"{passed / len(tasks) * 100:.1f}% success")
    print("replace RuleBasedAgent with your own model loop to benchmark it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
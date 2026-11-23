"""The episode loop: one task, one policy, one fresh world.

The runner is deliberately dumb and strict:

1. Build a fresh World from the task definition (never reuse one).
2. Ask the adapter for an action.
3. Route tool calls through the real tool layer, so schema validation, sandbox
   escapes and injected faults all apply.
4. Stop on 'finish', on step exhaustion, on repeated identical calls or on a
   sandbox escape.
5. Score the episode with the task's graders.

Every stop reason is surfaced in the results file, because "the agent ran out of
steps" and "the agent claimed to be done" are very different failures.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .adapters import Adapter, Observation
from .errors import AdapterError
from .grader import GraderRunner, score, summarise
from .models import CallRecord, EpisodeRecord, Task
from .tools import call_tool
from .world import ToolResult, World, normalise_path

STOP_FINISH = "finish"
STOP_MAX_STEPS = "max_steps"
STOP_REPEAT = "repeated_call"
STOP_ESCAPE = "sandbox_escape"
STOP_PROTOCOL = "protocol_error"


def _arg_signature(tool: str, args: Any) -> str:
    blob = json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_episode(task: Task, adapter: Adapter, repeat: int = 0,
                max_repeats: int = 3) -> EpisodeRecord:
    """Run a single task against a single adapter and return the record."""

    world = World.from_spec(
        task.world, seed=task.seed, faults=[fault.to_dict() for fault in task.faults]
    )
    adapter.reset(task)
    history: list[ToolResult] = []
    calls: list[CallRecord] = []
    signatures: dict[str, int] = {}
    answer = ""
    claimed: list[str] = []
    stop_reason = STOP_MAX_STEPS
    step = 0

    while step < task.max_steps:
        observation = Observation(task, step, history[-1] if history else None, history)
        try:
            action = adapter.next_action(observation)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"adapter {adapter.name!r} raised {type(exc).__name__} on task {task.id!r}: {exc}"
            ) from exc
        step += 1

        if not isinstance(action, dict):
            stop_reason = STOP_PROTOCOL
            answer = f"adapter returned {type(action).__name__}, expected an object"
            break
        kind = action.get("type")
        if kind == "finish":
            answer = str(action.get("answer", ""))
            claimed = [str(tool) for tool in (action.get("claimed_tools") or [])]
            stop_reason = STOP_FINISH
            break
        if kind != "tool":
            stop_reason = STOP_PROTOCOL
            answer = f"adapter returned unknown action type {kind!r}"
            break

        tool = str(action.get("tool", ""))
        raw_args = action.get("args")
        signature = _arg_signature(tool, raw_args)
        signatures[signature] = signatures.get(signature, 0) + 1
        if signatures[signature] > max_repeats:
            stop_reason = STOP_REPEAT
            answer = f"adapter repeated the same {tool} call more than {max_repeats} times"
            break

        try:
            result = call_tool(world, tool, raw_args)
        except Exception as exc:
            if type(exc).__name__ == "SandboxError":
                stop_reason = STOP_ESCAPE
                answer = f"adapter tried to escape the sandbox: {exc}"
                break
            raise
        history.append(result)
        if result.error_kind == "sandbox_escape":
            stop_reason = STOP_ESCAPE
            answer = f"adapter tried to escape the sandbox: {result.error}"
            break
        calls.append(
            CallRecord(
                index=result.call_index,
                tool=result.tool,
                args=dict(result.args),
                ok=result.ok,
                output=result.output,
                error=result.error,
                error_kind=result.error_kind,
                fault_injected=result.fault_injected,
                ticks=result.ticks,
            )
        )
    else:
        answer = ""

    outcomes = GraderRunner(world).run(task.graders, calls, answer, claimed)
    injected = sum(1 for call in calls if call.fault_injected)
    recovered = _count_recoveries(calls)
    value = score(outcomes)

    return EpisodeRecord(
        task_id=task.id,
        category=task.category,
        repeat=repeat,
        policy=adapter.name,
        answer=answer,
        stop_reason=stop_reason,
        score=value,
        passed=bool(outcomes) and all(outcome.passed for outcome in outcomes),
        steps=len(calls),
        calls=calls,
        checks=outcomes,
        recovered_faults=recovered,
        injected_faults=injected,
        claimed_tools=claimed,
    )


def _count_recoveries(calls: list[CallRecord]) -> int:
    """Count injected faults that were later followed by a success of the same tool."""

    recovered = 0
    for index, call in enumerate(calls):
        if not call.fault_injected:
            continue
        for later in calls[index + 1 :]:
            if later.tool == call.tool and later.ok:
                recovered += 1
                break
    return recovered


def _sandbox_guard(path: str) -> None:
    """Kept for adapters that want to validate a path before calling a tool."""

    normalise_path(path)


def check_answer_grounding(task: Task, record: EpisodeRecord) -> list[str]:
    """Convenience helper used by the CLI to explain why a task failed."""

    return [outcome.label for outcome in record.checks if not outcome.passed]


def summarise_episode(record: EpisodeRecord) -> str:
    return summarise(record.checks) if record.checks else "no graders ran"
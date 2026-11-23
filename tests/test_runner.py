"""The episode loop: stopping rules, faults and determinism."""

from __future__ import annotations

import pytest

from agent_bench.adapters import ScriptedAdapter
from agent_bench.errors import AdapterError
from agent_bench.models import Task
from agent_bench.runner import run_episode
from agent_bench.suite import parse_task


def make_task(**overrides) -> Task:
    payload = {
        "id": "t1",
        "category": "tool-selection",
        "title": "Read it",
        "instruction": "Read /work/a.txt and report the content verbatim.",
        "intent": {
            "steps": [{"id": "s1", "tool": "fs_read", "args": {"path": "/work/a.txt"}}],
            "answer": "the file says {s1.content}",
        },
        "world": {"files": {"/work/a.txt": "hello"}},
        "graders": [
            {"kind": "answer_contains", "weight": 1.0,
             "description": "mentions the content", "params": {"text": "hello"}},
            {"kind": "tool_called", "weight": 1.0,
             "description": "read the file", "params": {"tool": "fs_read"}},
        ],
    }
    payload.update(overrides)
    return parse_task(payload)


class Greedy:
    """Always calls a tool and never finishes."""

    name = "greedy"

    def __init__(self) -> None:
        self.n = 0

    def reset(self, task: Task) -> None:
        self.n = 0

    def next_action(self, observation) -> dict:
        self.n += 1
        return {"type": "tool", "tool": "fs_read", "args": {"path": f"/work/a.txt?{self.n}"}}


class Repeater:
    """Calls the exact same tool with the exact same args forever."""

    name = "repeater"

    def reset(self, task: Task) -> None:
        return None

    def next_action(self, observation) -> dict:
        return {"type": "tool", "tool": "fs_read", "args": {"path": "/work/a.txt"}}


class Escaper:
    """Tries to read outside the sandbox."""

    name = "escaper"

    def reset(self, task: Task) -> None:
        return None

    def next_action(self, observation) -> dict:
        return {"type": "tool", "tool": "fs_read", "args": {"path": "../../etc/passwd"}}


class ProtocolBreaker:
    name = "protocol"

    def reset(self, task: Task) -> None:
        return None

    def next_action(self, observation) -> dict:
        return {"type": "run_shell", "tool": "fs_read"}


class Crasher:
    name = "crasher"

    def reset(self, task: Task) -> None:
        return None

    def next_action(self, observation) -> dict:
        raise RuntimeError("model exploded")


def test_scripted_adapter_passes_the_happy_path() -> None:
    record = run_episode(make_task(), ScriptedAdapter(persona="careful"))
    assert record.passed
    assert record.score == 1.0
    assert record.stop_reason == "finish"
    assert len(record.calls) == 1


def test_greedy_policy_stops_at_max_steps() -> None:
    record = run_episode(make_task(max_steps=3), Greedy())
    assert record.stop_reason == "max_steps"
    assert len(record.calls) == 3
    assert not record.passed


def test_identical_calls_are_cut_off() -> None:
    record = run_episode(make_task(max_steps=20), Repeater(), max_repeats=2)
    assert record.stop_reason == "repeated_call"


def test_sandbox_escape_stops_the_episode() -> None:
    record = run_episode(make_task(), Escaper())
    assert record.stop_reason == "sandbox_escape"
    assert not record.passed


def test_unknown_action_type_is_a_protocol_error() -> None:
    record = run_episode(make_task(), ProtocolBreaker())
    assert record.stop_reason == "protocol_error"
    assert "run_shell" in record.answer


def test_adapter_exceptions_become_adapter_errors() -> None:
    with pytest.raises(AdapterError):
        run_episode(make_task(), Crasher())


def test_injected_fault_is_retried_by_the_careful_persona() -> None:
    task = make_task(faults=[{"tool": "fs_read", "kind": "timeout", "times": 1}])
    record = run_episode(task, ScriptedAdapter(persona="careful"))
    assert record.injected_faults == 1
    assert record.recovered_faults == 1
    assert record.passed


def test_naive_persona_gives_up_on_the_fault() -> None:
    task = make_task(faults=[{"tool": "fs_read", "kind": "timeout", "times": 1}])
    record = run_episode(task, ScriptedAdapter(persona="naive"))
    assert record.injected_faults == 1
    assert record.recovered_faults == 0
    assert not record.passed


def test_hasty_persona_plans_correctly_but_does_not_recover() -> None:
    task = make_task(faults=[{"tool": "fs_read", "kind": "timeout", "times": 1}])
    record = run_episode(task, ScriptedAdapter(persona="hasty"))
    assert record.stop_reason == "finish"
    assert record.recovered_faults == 0


def test_episodes_are_deterministic() -> None:
    task = make_task()
    first = run_episode(task, ScriptedAdapter(persona="careful")).to_dict()
    second = run_episode(task, ScriptedAdapter(persona="careful")).to_dict()
    assert first == second


def test_worlds_do_not_leak_between_episodes() -> None:
    task = make_task()
    adapter = ScriptedAdapter(persona="careful")
    first = run_episode(task, adapter)
    second = run_episode(task, adapter)
    assert first.to_dict() == second.to_dict()


def test_task_without_steps_reports_adapter_error() -> None:
    from agent_bench.models import Task

    task = Task(id="x", category="tool-selection", title="t", instruction="do something useful",
                graders=[], intent={})
    with pytest.raises(AdapterError):
        run_episode(task, ScriptedAdapter())

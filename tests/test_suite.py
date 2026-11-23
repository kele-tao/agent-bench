"""Suite loading, validation and scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_bench.errors import SuiteError
from agent_bench.models import CATEGORIES
from agent_bench.suite import (
    discover_suites,
    load_bundled_suite,
    load_suite,
    parse_suite,
    resolve_suite,
    scaffold_suite,
    scaffold_task,
)


def minimal_task(**overrides) -> dict:
    task = {
        "id": "t1",
        "category": "tool-selection",
        "title": "Read a file",
        "instruction": "Read /work/a.txt and report it.",
        "intent": {"steps": [{"id": "s1", "tool": "fs_read", "args": {"path": "/work/a.txt"}}],
                   "answer": "{s1.content}"},
        "world": {"files": {"/work/a.txt": "hi"}},
        "graders": [{"kind": "answer_contains", "weight": 1.0,
                     "description": "mentions the file body", "params": {"text": "hi"}}],
    }
    task.update(overrides)
    return task


def write_suite(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_bundled_core_has_twelve_tasks() -> None:
    name, tasks = load_bundled_suite("core")
    assert name == "core"
    assert len(tasks) == 12


def test_bundled_hard_has_five_tasks() -> None:
    _, tasks = load_bundled_suite("hard")
    assert len(tasks) == 5


def test_bundled_suites_cover_every_category() -> None:
    categories = set()
    for suite in discover_suites():
        categories |= {task.category for task in suite["tasks"]}
    assert categories == set(CATEGORIES)


def test_every_bundled_task_has_graders_faults_and_intent() -> None:
    for suite in discover_suites():
        for task in suite["tasks"]:
            assert task.graders, task.id
            assert task.intent.get("steps"), task.id
            assert task.max_steps >= 1
            assert task.notes, f"{task.id} should explain the failure mode it exposes"


def test_bundled_suites_are_deterministic() -> None:
    first = [task.to_dict() for task in load_bundled_suite("core")[1]]
    second = [task.to_dict() for task in load_bundled_suite("core")[1]]
    assert first == second


def test_load_suite_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SuiteError):
        load_suite(tmp_path / "nope.json")


def test_load_suite_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(SuiteError) as excinfo:
        load_suite(path)
    assert "not valid JSON" in str(excinfo.value)


def test_parse_suite_rejects_empty_tasks() -> None:
    with pytest.raises(SuiteError):
        parse_suite({"name": "x", "tasks": []})


def test_parse_suite_rejects_duplicate_ids() -> None:
    payload = {"name": "x", "tasks": [minimal_task(), minimal_task()]}
    with pytest.raises(SuiteError) as excinfo:
        parse_suite(payload)
    assert "duplicate" in str(excinfo.value)


@pytest.mark.parametrize("field", ["id", "category", "title", "instruction", "graders"])
def test_parse_task_requires_core_fields(field: str) -> None:
    task = minimal_task()
    del task[field]
    with pytest.raises(SuiteError):
        parse_suite({"name": "x", "tasks": [task]})


def test_parse_task_rejects_unknown_category() -> None:
    with pytest.raises(SuiteError):
        parse_suite({"name": "x", "tasks": [minimal_task(category="vibes")]})


def test_parse_task_rejects_unknown_grader_kind() -> None:
    task = minimal_task()
    task["graders"][0]["kind"] = "vibes_check"
    with pytest.raises(SuiteError) as excinfo:
        parse_suite({"name": "x", "tasks": [task]})
    assert "unknown grader kind" in str(excinfo.value)


def test_parse_task_rejects_fault_on_unknown_tool() -> None:
    task = minimal_task(faults=[{"tool": "teleport", "kind": "timeout"}])
    with pytest.raises(SuiteError):
        parse_suite({"name": "x", "tasks": [task]})


def test_parse_task_rejects_malformed_json_fault_on_text_tool() -> None:
    task = minimal_task(faults=[{"tool": "calc", "kind": "malformed_json"}])
    with pytest.raises(SuiteError):
        parse_suite({"name": "x", "tasks": [task]})


def test_parse_task_requires_intent_steps() -> None:
    task = minimal_task()
    del task["intent"]
    with pytest.raises(SuiteError):
        parse_suite({"name": "x", "tasks": [task]})


def test_parse_task_rejects_sandbox_escaping_world_path() -> None:
    task = minimal_task(world={"files": {"../escape.txt": "x"}})
    with pytest.raises(SuiteError):
        parse_suite({"name": "x", "tasks": [task]})


def test_parse_task_accepts_unicode_instruction() -> None:
    task = minimal_task(instruction="读取 /work/a.txt 并报告内容 U0001f600 please")
    _, tasks = parse_suite({"name": "x", "tasks": [task]})
    assert "U0001f600" in tasks[0].instruction


def test_scaffold_task_round_trips_through_the_loader() -> None:
    task = scaffold_task("my-task", "tool-selection")
    suite = {"name": "custom", "tasks": [task]}
    _, tasks = parse_suite(suite)
    assert tasks[0].id == "my-task"


def test_scaffold_task_rejects_unknown_category() -> None:
    with pytest.raises(SuiteError):
        scaffold_task("x", "nonsense")


def test_scaffold_suite_contains_one_task() -> None:
    payload = scaffold_suite("demo", "error-recovery")
    assert payload["tasks"][0]["category"] == "error-recovery"


def test_resolve_suite_accepts_a_path(tmp_path: Path) -> None:
    path = write_suite(tmp_path, {"name": "custom", "tasks": [minimal_task()]})
    name, tasks = resolve_suite(str(path))
    assert name == "custom" and len(tasks) == 1


def test_resolve_suite_rejects_unknown_bundled_name() -> None:
    with pytest.raises(SuiteError) as excinfo:
        resolve_suite("nope")
    assert "bundled suites" in str(excinfo.value.hint)
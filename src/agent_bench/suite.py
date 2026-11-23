"""Suite loading, validation and scaffolding.

A suite is a JSON document with a name, a version, a description and a tasks
array. Validation is strict and eager: a suite with a typo in a grader kind, a
fault pointing at a tool that does not exist, or a task the scripted adapter
cannot plan fails before a single episode runs. Bad input must never produce a
half-populated leaderboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import SuiteError
from .models import CATEGORIES, FaultSpec, Grader, Task
from .tools import MOCK_TOOLS, TOOL_SCHEMAS
from .world import FAULT_KINDS, JSON_TOOLS, validate_fault

REQUIRED_TASK_FIELDS = ("id", "category", "title", "instruction", "graders")
TASK_TEMPLATE = {
    "id": "my-task",
    "category": "tool-selection",
    "title": "Short imperative title",
    "instruction": "What the agent must do, phrased as an operator would phrase it.",
    "weight": 1.0,
    "max_steps": 8,
    "seed": 0,
    "intent": {
        "steps": [{"id": "s1", "tool": "fs_read", "args": {"path": "/work/notes.txt"}}],
        "answer": "The answer is {s1}.",
        "naive": {
            "skip": ["s1"],
            "answer": "I think the answer is 42.",
            "claim": ["fs_read"],
        },
    },
    "world": {"files": {"/work/notes.txt": "answer: 42"}},
    "faults": [],
    "graders": [
        {
            "kind": "answer_contains",
            "weight": 1.0,
            "description": "answer states the retrieved value",
            "params": {"text": "42"},
        }
    ],
    "notes": "Explain the failure mode this task exposes.",
}


def _fail(message: str, hint: str | None = None) -> SuiteError:
    return SuiteError(message, hint=hint)


def load_suite(path: str | Path) -> tuple[str, list[Task]]:
    """Read a suite file and return (suite_name, tasks)."""

    file_path = Path(path)
    if not file_path.exists():
        raise _fail(
            f"suite file not found: {file_path}",
            hint="run 'agent-bench list' to see the bundled suites",
        )
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise _fail(f"suite is not UTF-8: {file_path}") from exc
    except ValueError as exc:
        raise _fail(
            f"suite is not valid JSON: {file_path} ({exc.msg} at line {exc.lineno})",
            hint="check for a trailing comma or a missing brace",
        ) from exc
    return parse_suite(raw, source=str(file_path))


def parse_suite(raw: Any, source: str = "<memory>") -> tuple[str, list[Task]]:
    """Validate an already decoded suite document."""

    if not isinstance(raw, dict):
        raise _fail(f"{source}: a suite must be a JSON object")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _fail(f"{source}: suite needs a non-empty 'name'")
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise _fail(f"{source}: suite {name!r} needs a non-empty 'tasks' array")

    tasks: list[Task] = []
    seen: set[str] = set()
    for index, entry in enumerate(tasks_raw):
        task = parse_task(entry, suite=name, source=f"{source}[{index}]")
        if task.id in seen:
            raise _fail(f"{source}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)
    return name, tasks

def parse_task(entry: Any, suite: str = "", source: str = "<task>") -> Task:
    """Validate one task definition."""

    if not isinstance(entry, dict):
        raise _fail(f"{source}: a task must be a JSON object")
    for field in REQUIRED_TASK_FIELDS:
        if field not in entry:
            raise _fail(
                f"{source}: task is missing required field {field!r}",
                hint="run 'agent-bench new-task' to see a complete template",
            )
    task_id = entry["id"]
    if not isinstance(task_id, str) or not task_id.strip():
        raise _fail(f"{source}: task 'id' must be a non-empty string")
    if not isinstance(entry["instruction"], str) or len(entry["instruction"].strip()) < 10:
        raise _fail(f"{source}: task {task_id!r} needs an instruction of at least 10 characters")
    category = entry["category"]
    if category not in CATEGORIES:
        raise _fail(
            f"{source}: task {task_id!r} has unknown category {category!r}",
            hint="valid categories: " + ", ".join(CATEGORIES),
        )
    weight = entry.get("weight", 1.0)
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
        raise _fail(f"{source}: task {task_id!r} weight must be a positive number")
    max_steps = entry.get("max_steps", 10)
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 50:
        raise _fail(f"{source}: task {task_id!r} max_steps must be an integer 1..50")
    seed = entry.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise _fail(f"{source}: task {task_id!r} seed must be an integer")

    graders_raw = entry["graders"]
    if not isinstance(graders_raw, list) or not graders_raw:
        raise _fail(
            f"{source}: task {task_id!r} needs at least one grader",
            hint="a task with no grader measures nothing",
        )
    graders = [_parse_grader(item, task_id, source) for item in graders_raw]

    world = entry.get("world", {})
    if not isinstance(world, dict):
        raise _fail(f"{source}: task {task_id!r} world must be an object")
    _validate_world(task_id, world, source)

    faults: list[FaultSpec] = []
    faults_raw = entry.get("faults", [])
    if not isinstance(faults_raw, list):
        raise _fail(f"{source}: task {task_id!r} faults must be an array")
    for fault in faults_raw:
        if not isinstance(fault, dict):
            raise _fail(f"{source}: task {task_id!r} fault entries must be objects")
        validate_fault(fault, set(MOCK_TOOLS))
        times = fault.get("times", 1)
        if not isinstance(times, int) or isinstance(times, bool) or not 1 <= times <= 5:
            raise _fail(f"{source}: task {task_id!r} fault 'times' must be an integer 1..5")
        faults.append(FaultSpec(tool=str(fault["tool"]), kind=str(fault["kind"]), times=times))

    intent = entry.get("intent", {})
    if not isinstance(intent, dict):
        raise _fail(f"{source}: task {task_id!r} intent must be an object")
    _validate_intent(task_id, intent, source)

    return Task(
        id=task_id,
        category=category,
        title=str(entry["title"]),
        instruction=entry["instruction"].strip(),
        weight=float(weight),
        max_steps=max_steps,
        seed=seed,
        intent=intent,
        world=world,
        faults=faults,
        graders=graders,
        notes=str(entry.get("notes", "")),
        suite=suite,
    )

def _parse_grader(item: Any, task_id: str, source: str) -> Grader:
    if not isinstance(item, dict):
        raise _fail(f"{source}: task {task_id!r} grader entries must be objects")
    kind = item.get("kind")
    if not isinstance(kind, str) or not kind:
        raise _fail(f"{source}: task {task_id!r} grader needs a 'kind'")
    from .grader import STRICT_GRADERS

    if kind not in STRICT_GRADERS:
        raise _fail(
            f"{source}: task {task_id!r} uses unknown grader kind {kind!r}",
            hint="valid kinds: " + ", ".join(sorted(STRICT_GRADERS)),
        )
    weight = item.get("weight", 1.0)
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
        raise _fail(f"{source}: task {task_id!r} grader weight must be a positive number")
    params = item.get("params", {})
    if not isinstance(params, dict):
        raise _fail(f"{source}: task {task_id!r} grader params must be an object")
    description = item.get("description")
    if not isinstance(description, str) or not description.strip():
        raise _fail(
            f"{source}: task {task_id!r} grader needs a human readable 'description'",
            hint="the description is what the report shows when the check fails",
        )
    _validate_grader_params(task_id, kind, params, source)
    return Grader(kind=kind, params=params, weight=float(weight), description=description.strip())


GRADER_PARAMS: dict[str, tuple[str, ...]] = {
    "answer_contains": ("text",),
    "answer_excludes": ("text",),
    "answer_matches": ("pattern",),
    "tool_called": ("tool",),
    "tool_not_called": ("tool",),
    "file_exists": ("path",),
    "file_contains": ("path",),
    "file_absent": ("path",),
    "used_tools": ("tools",),
    "max_steps": ("limit",),
    "max_calls_of_tool": ("tool", "limit"),
    "recovered": ("tool",),
    "answer_max_words": ("limit",),
    "email_sent": (),
    "email_not_sent": (),
    "event_created": (),
    "no_side_effects": (),
    "claimed_tool_honest": (),
    "numbers_grounded": (),
    "answer_not_empty": (),
}


def _validate_grader_params(task_id: str, kind: str, params: dict[str, Any], source: str) -> None:
    for field in GRADER_PARAMS.get(kind, ()):
        if field not in params:
            raise _fail(f"{source}: task {task_id!r} grader {kind!r} needs params.{field}")
    tool_scoped = ("tool_called", "tool_not_called", "max_calls_of_tool")
    if kind in tool_scoped and params["tool"] not in TOOL_SCHEMAS:
        raise _fail(
            f"{source}: task {task_id!r} grader {kind!r} references unknown tool "
            f"{params['tool']!r}"
        )
    if kind == "used_tools":
        unknown = [tool for tool in params["tools"] if tool not in TOOL_SCHEMAS]
        if unknown:
            raise _fail(
                f"{source}: task {task_id!r} used_tools references unknown tool(s): "
                + ", ".join(map(str, unknown))
            )
    if kind == "answer_matches":
        import re

        try:
            re.compile(params["pattern"])
        except re.error as exc:
            raise _fail(
                f"{source}: task {task_id!r} grader answer_matches has an invalid regex: {exc}"
            ) from exc

def _validate_world(task_id: str, world: dict[str, Any], source: str) -> None:
    from .world import normalise_path

    files = world.get("files", {})
    if not isinstance(files, dict):
        raise _fail(f"{source}: task {task_id!r} world.files must be an object")
    for path in files:
        try:
            normalise_path(path)
        except Exception as exc:
            raise _fail(
                f"{source}: task {task_id!r} has an invalid world path {path!r}: {exc}"
            ) from exc
    for section in ("http", "tables"):
        if not isinstance(world.get(section, {}), dict):
            raise _fail(f"{source}: task {task_id!r} world.{section} must be an object")
    if not isinstance(world.get("docs", []), list):
        raise _fail(f"{source}: task {task_id!r} world.docs must be an array")
    if not isinstance(world.get("clarifications", []), list):
        raise _fail(f"{source}: task {task_id!r} world.clarifications must be an array")
    if not isinstance(world.get("email_policy", {}), dict):
        raise _fail(f"{source}: task {task_id!r} world.email_policy must be an object")


def _validate_intent(task_id: str, intent: dict[str, Any], source: str) -> None:
    steps = intent.get("steps")
    if steps is None:
        raise _fail(
            f"{source}: task {task_id!r} needs intent.steps so the scripted adapter can plan"
        )
    if not isinstance(steps, list) or not steps:
        raise _fail(f"{source}: task {task_id!r} intent.steps must be a non-empty array")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise _fail(f"{source}: task {task_id!r} intent.steps[{index}] must be an object")
        tool = step.get("tool")
        if tool not in MOCK_TOOLS:
            raise _fail(
                f"{source}: task {task_id!r} intent.steps[{index}] uses unknown tool {tool!r}",
                hint="scripted steps can only use mock tools, not 'finish'",
            )
        if not isinstance(step.get("args", {}), dict):
            raise _fail(f"{source}: task {task_id!r} intent.steps[{index}].args must be an object")
    naive = intent.get("naive")
    if naive is not None and not isinstance(naive, dict):
        raise _fail(f"{source}: task {task_id!r} intent.naive must be an object")


def scaffold_task(task_id: str, category: str) -> dict[str, Any]:
    """Return a complete, runnable starter task."""

    if category not in CATEGORIES:
        raise _fail(
            f"unknown category {category!r}",
            hint="valid categories: " + ", ".join(CATEGORIES),
        )
    task = json.loads(json.dumps(TASK_TEMPLATE))
    task["id"] = task_id
    task["category"] = category
    return task


def scaffold_suite(task_id: str, category: str) -> dict[str, Any]:
    """Wrap scaffold_task in a one-task suite."""

    return {
        "name": "custom",
        "version": 1,
        "description": "A suite scaffolded by agent-bench new-task.",
        "tasks": [scaffold_task(task_id, category)],
    }


def suites_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "suites"


def load_bundled_suite(name: str) -> tuple[str, list[Task]]:
    """Load a suite by bare name, e.g. 'core' or 'hard'."""

    candidate = suites_dir() / f"{name}.json"
    if candidate.exists():
        return load_suite(candidate)
    raise _fail(
        f"no bundled suite named {name!r}",
        hint="bundled suites: core, hard; or pass a path to your own suite.json",
    )


def discover_suites() -> list[dict[str, Any]]:
    """Summarise every bundled suite, for the 'list' command."""

    found: list[dict[str, Any]] = []
    for path in sorted(suites_dir().glob("*.json")):
        name, tasks = load_suite(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        found.append(
            {
                "name": name,
                "path": "suites/" + path.name,
                "description": str(meta.get("description", "")),
                "tasks": tasks,
            }
        )
    return found


def resolve_suite(spec: str) -> tuple[str, list[Task]]:
    """Resolve a CLI --suite value: a bundled name or a path to a JSON file."""

    if spec.endswith(".json") or "/" in spec or "\\" in spec:
        return load_suite(spec)
    return load_bundled_suite(spec)


def validate_tool_catalogue() -> None:
    """Cheap internal consistency check run before every CLI invocation."""

    missing = [tool for tool in MOCK_TOOLS if tool not in TOOL_SCHEMAS]
    if missing:
        raise SuiteError("internal error: tools without schema: " + ", ".join(missing))
    unsupported = sorted(set(FAULT_KINDS) - {"timeout", "http_500", "malformed_json"})
    if unsupported:
        raise SuiteError("internal error: unsupported fault kinds: " + ", ".join(unsupported))
    orphan = sorted(JSON_TOOLS - set(TOOL_SCHEMAS))
    if orphan:
        raise SuiteError("internal error: JSON tools without schema: " + ", ".join(orphan))
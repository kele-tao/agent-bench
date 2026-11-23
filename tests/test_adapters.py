"""Adapter protocol, template resolution and the adapter factory."""

from __future__ import annotations

import pytest

from agent_bench.adapters import (
    PERSONAS,
    Observation,
    ScriptedAdapter,
    TemplateError,
    build_adapter,
    resolve,
)
from agent_bench.errors import AdapterError
from agent_bench.models import Task


def test_resolve_plain_literal() -> None:
    assert resolve("no placeholders", {}) == "no placeholders"


def test_resolve_nested_field() -> None:
    results = {"s1": {"body": {"state": "degraded"}}}
    assert resolve("state is {s1.body.state}", results) == "state is degraded"


def test_resolve_index_into_list() -> None:
    results = {"s1": {"rows": [{"email": "ada@example.com"}]}}
    assert resolve("{s1.rows[0].email}", results) == "ada@example.com"


def test_resolve_numbers_keep_their_text() -> None:
    assert resolve("{s1.result}", {"s1": {"result": "472.084"}}) == "472.084"
    assert resolve("{s1.count}", {"s1": {"count": 3}}) == "3"


def test_resolve_unknown_step_raises() -> None:
    with pytest.raises(TemplateError):
        resolve("{nope.field}", {})


def test_resolve_bad_index_raises() -> None:
    with pytest.raises(TemplateError):
        resolve("{s1.rows[9].email}", {"s1": {"rows": [{"email": "x"}]}})


def test_resolve_rejects_format_specs() -> None:
    with pytest.raises(TemplateError):
        resolve("{s1:>10}", {"s1": "x"})


def test_unknown_persona_is_a_friendly_error() -> None:
    with pytest.raises(AdapterError) as excinfo:
        ScriptedAdapter(persona="chaotic")
    assert "careful" in str(excinfo.value.hint)


def test_personas_are_declared() -> None:
    assert set(PERSONAS) == {"careful", "hasty", "naive"}


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("scripted", "scripted"), ("naive", "scripted-naive"), ("hasty", "scripted-hasty")],
)
def test_build_adapter_maps_names(kind: str, expected: str) -> None:
    assert build_adapter(kind).name == expected


def test_build_adapter_rejects_unknown_kind() -> None:
    with pytest.raises(AdapterError) as excinfo:
        build_adapter("gpt5-vibes")
    assert "choices" in str(excinfo.value.hint)


def test_openai_adapter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_BENCH_API_KEY", raising=False)
    with pytest.raises(AdapterError) as excinfo:
        build_adapter("openai")
    assert "AGENT_BENCH_API_KEY" in str(excinfo.value)


def test_openai_adapter_parses_tool_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_BENCH_API_KEY", "test-key-not-real")
    adapter = build_adapter("openai")
    action = adapter._parse('I will look. {"tool": "fs_read", "args": {"path": "/work/a"}}')
    assert action["type"] == "tool"
    assert action["tool"] == "fs_read"


def test_openai_adapter_parses_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_BENCH_API_KEY", "test-key-not-real")
    adapter = build_adapter("openai")
    action = adapter._parse('{"answer": "done", "claimed_tools": ["fs_read"]}')
    assert action["type"] == "finish"
    assert action["claimed_tools"] == ["fs_read"]


def test_openai_adapter_never_logs_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_BENCH_API_KEY", "super-secret-value")
    adapter = build_adapter("openai")
    assert adapter.api_key == "super-secret-value"
    assert "super-secret-value" not in adapter.name
    assert "super-secret-value" not in adapter._schema_text


def test_observation_transcript_includes_errors() -> None:
    from agent_bench.world import ToolResult

    task = Task(id="t", category="tool-selection", title="t", instruction="do it", graders=[],
                intent={"steps": [{"tool": "fs_read", "args": {}}]})
    failure = ToolResult(tool="fs_read", args={}, ok=False, error="boom",
                         error_kind="timeout", fault_injected=True)
    observation = Observation(task, 1, failure, [failure])
    text = observation.history_text()
    assert "boom" in text
    assert "timeout" in text
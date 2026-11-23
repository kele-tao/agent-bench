"""Tool schemas, calculator and fault behaviour."""

from __future__ import annotations

import json

import pytest

from agent_bench.tools import (
    MOCK_TOOLS,
    TOOL_SCHEMAS,
    call_tool,
    evaluate,
    parse_raw_args,
    tool_catalogue,
    validate_args,
)
from agent_bench.world import ToolFailure, World


def make_world(**spec) -> World:
    return World.from_spec(spec)


def test_every_mock_tool_has_a_handler_and_schema() -> None:
    assert len(MOCK_TOOLS) >= 8
    for tool in MOCK_TOOLS:
        assert tool in TOOL_SCHEMAS


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 1", "2"),
        ("2 * 3 + 4", "10"),
        ("10 / 4", "2.5"),
        ("(1234.56 * 1.075 + 89.10) / 3", "472.084"),
        ("round(2.345, 2)", "2.34"),
        ("max(1, 9) - min(3, 4)", "6"),
        ("2 ** 10", "1024"),
    ],
)
def test_evaluate_exact_arithmetic(expression: str, expected: str) -> None:
    assert evaluate(expression) == expected


@pytest.mark.parametrize("expression", ["", "   ", "1 / 0", "1 +", "__import__('os')", "abc"])
def test_evaluate_rejects_bad_expressions(expression: str) -> None:
    with pytest.raises(ToolFailure):
        evaluate(expression)


def test_evaluate_rejects_huge_exponents() -> None:
    with pytest.raises(ToolFailure):
        evaluate("2 ** 99")


def test_evaluate_rejects_long_expression() -> None:
    with pytest.raises(ToolFailure):
        evaluate("1+" * 200 + "1")


def test_validate_args_reports_every_problem() -> None:
    errors = validate_args(TOOL_SCHEMAS["fs_read"], {"path": "", "extra": 1})
    assert any("minLength" in error or "at least" in error for error in errors)
    assert any("unexpected argument" in error for error in errors)


def test_validate_args_type_error() -> None:
    errors = validate_args(TOOL_SCHEMAS["search_docs"], {"query": 5})
    assert errors and "expected string" in errors[0]


def test_validate_args_requires_fields() -> None:
    errors = validate_args(TOOL_SCHEMAS["send_email"], {"to": "a@b.c"})
    assert any("subject" in error for error in errors)


def test_parse_raw_args_accepts_json_string() -> None:
    assert parse_raw_args('{"path": "/work/a"}') == ({"path": "/work/a"}, None)


@pytest.mark.parametrize("raw", ["{not json", "[]", 5])
def test_parse_raw_args_rejects_garbage(raw: object) -> None:
    args, error = parse_raw_args(raw)
    assert args is None and error


def test_call_tool_happy_path_reads_file() -> None:
    world = make_world(files={"/work/a.txt": "hello"})
    result = call_tool(world, "fs_read", {"path": "/work/a.txt"})
    assert result.ok
    assert json.loads(result.output)["content"] == "hello"
    assert world.calls[-1].call_index == 1


def test_call_tool_records_schema_violation_without_executing() -> None:
    world = make_world(files={"/work/a.txt": "hello"})
    result = call_tool(world, "fs_read", {})
    assert not result.ok
    assert result.error_kind == "schema_violation"
    assert "required argument" in result.error


def test_call_tool_unknown_tool_lists_alternatives() -> None:
    result = call_tool(make_world(), "frobnicate", {})
    assert not result.ok
    assert result.error_kind == "unknown_tool"
    assert "fs_read" in result.error


def test_call_tool_injected_timeout_has_no_output() -> None:
    world = make_world(files={"/work/a.txt": "x"}, faults=[{"tool": "fs_read", "kind": "timeout"}])
    result = call_tool(world, "fs_read", {"path": "/work/a.txt"})
    assert not result.ok and result.fault_injected
    assert result.error_kind == "timeout"
    assert result.output is None


def test_call_tool_malformed_json_is_unparseable() -> None:
    world = make_world(faults=[{"tool": "search_docs", "kind": "malformed_json"}])
    result = call_tool(world, "search_docs", {"query": "anything"})
    assert not result.ok
    assert result.error_kind == "malformed_json"
    assert result.fault_injected
    assert result.payload() is None


def test_call_tool_sandbox_escape_is_recorded() -> None:
    result = call_tool(make_world(), "fs_read", {"path": "../../etc/passwd"})
    assert not result.ok
    assert result.error_kind == "sandbox_escape"


def test_calc_via_tool_returns_json_payload() -> None:
    result = call_tool(make_world(), "calc", {"expression": "6 * 7"})
    assert result.ok
    assert result.payload()["result"] == "42"


def test_http_get_500_is_a_tool_failure() -> None:
    world = make_world(http={"/boom": {"status": 500}})
    result = call_tool(world, "http_get", {"url": "/boom"})
    assert not result.ok
    assert result.error_kind == "http_500"


def test_http_post_returns_route_body() -> None:
    world = make_world(http={"/orders": {"post_body": {"order_id": "ord-1"}}})
    result = call_tool(world, "http_post", {"url": "/orders", "body": {"qty": 2}})
    payload = result.payload()
    assert payload["body"]["order_id"] == "ord-1"
    assert payload["received"] == {"qty": 2}


def test_tool_catalogue_marks_finish_as_terminal() -> None:
    catalogue = {entry["name"]: entry for entry in tool_catalogue()}
    assert catalogue["finish"]["kind"] == "terminal"
    assert catalogue["fs_read"]["kind"] == "mock"


def test_wasted_call_counter_matches_failures() -> None:
    world = make_world(files={"/work/a.txt": "x"})
    call_tool(world, "fs_read", {"path": "/work/missing.txt"})
    call_tool(world, "fs_read", {"path": "/work/a.txt"})
    assert sum(1 for call in world.calls if not call.ok) == 1
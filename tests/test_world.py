"""World, sandbox and path handling."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agent_bench.errors import SuiteError
from agent_bench.world import (
    SandboxError,
    ToolFailure,
    World,
    format_decimal,
    normalise_path,
    validate_fault,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/work/a.txt", "/work/a.txt"),
        ("a.txt", "/work/a.txt"),
        ("work/a.txt", "/work/a.txt"),
        ("/work/", "/work"),
        ("/", "/work"),
        ("/work/./a/../b.txt", "/work/b.txt"),
        ("/work//a.txt", "/work/a.txt"),
        ("\\work\\a.txt", "/work/a.txt"),
    ],
)
def test_normalise_path_is_idempotent(path: str, expected: str) -> None:
    assert normalise_path(path) == expected
    assert normalise_path(normalise_path(path)) == expected


@pytest.mark.parametrize(
    "path",
    ["../etc/passwd", "/..", "~/secrets", "C:/Windows/system32", "", "   ", "a/b../c"],
)
def test_normalise_path_rejects_escapes(path: str) -> None:
    with pytest.raises(SandboxError):
        normalise_path(path)


def test_format_decimal_is_exact() -> None:
    assert format_decimal(Decimal("472.084")) == "472.084"
    assert format_decimal(Decimal("100.000")) == "100"
    assert format_decimal(Decimal("0.10")) == "0.1"
    assert format_decimal(Decimal("0")) == "0"


def test_world_files_are_independent_between_instances() -> None:
    spec = {"files": {"/work/a.txt": "one"}}
    first = World.from_spec(spec)
    second = World.from_spec(spec)
    first.write_file("/work/a.txt", "two")
    assert second.read_file("/work/a.txt") == "one"


def test_list_dir_reports_files_and_directories() -> None:
    world = World.from_spec({"files": {"/work/a.txt": "x", "/work/sub/b.txt": "yy"}})
    entries = world.list_dir("/work")
    by_name = {entry["name"]: entry for entry in entries}
    assert by_name["a.txt"]["type"] == "file"
    assert by_name["a.txt"]["size"] == 1
    assert by_name["sub"]["type"] == "directory"


def test_list_dir_missing_directory_raises() -> None:
    world = World.from_spec({"files": {"/work/a.txt": "x"}})
    with pytest.raises(ToolFailure) as excinfo:
        world.list_dir("/work/nope")
    assert excinfo.value.kind == "file_not_found"


def test_query_table_sorts_and_filters() -> None:
    world = World.from_spec(
        {
            "tables": {
                "orders": [
                    {"id": 2, "status": "pending"},
                    {"id": 1, "status": "pending"},
                    {"id": 3, "status": "paid"},
                ]
            }
        }
    )
    rows = world.query_table("orders", {"status": "pending"})
    assert [row["id"] for row in rows] == [1, 2]


def test_query_table_unknown_column_raises() -> None:
    world = World.from_spec({"tables": {"orders": [{"id": 1}]}})
    with pytest.raises(ToolFailure) as excinfo:
        world.query_table("orders", {"colour": "red"})
    assert excinfo.value.kind == "unknown_column"


def test_search_docs_ranks_title_matches_first() -> None:
    world = World.from_spec(
        {
            "docs": [
                {"title": "Refund policy 2026", "text": "refund window is 30 days"},
                {"title": "Shipping", "text": "refund mentions are rare here"},
            ]
        }
    )
    hits = world.search_docs("refund")
    assert hits[0]["title"] == "Refund policy 2026"


def test_search_docs_empty_query_raises() -> None:
    world = World.from_spec({"docs": []})
    with pytest.raises(ToolFailure):
        world.search_docs("   ")


def test_email_policy_denies_external_domain() -> None:
    world = World.from_spec({"email_policy": {"allow": ["@example.com"], "deny": ["@evil.test"]}})
    with pytest.raises(ToolFailure) as excinfo:
        world.send_email("a@evil.test", "hi", "body")
    assert excinfo.value.kind == "policy_denied"
    assert world.outbox == []


def test_email_policy_allows_listed_domain() -> None:
    world = World.from_spec({"email_policy": {"allow": ["@example.com"]}})
    world.send_email("a@example.com", "hi", "body")
    assert world.outbox[0]["to"] == "a@example.com"


def test_create_event_rejects_double_booking() -> None:
    world = World.from_spec({})
    world.create_event("standup", "2026-11-24 09:00")
    with pytest.raises(ToolFailure) as excinfo:
        world.create_event("other", "2026-11-24 09:00")
    assert excinfo.value.kind == "conflict"


def test_ask_user_matches_clarification() -> None:
    world = World.from_spec({"clarifications": [{"match": "environment", "answer": "staging"}]})
    assert world.ask_user("Which environment should I deploy to?")["answer"] == "staging"
    assert world.ask_user("what time is it")["matched"] is False


def test_validate_fault_rejects_unknown_tool() -> None:
    with pytest.raises(SuiteError):
        validate_fault({"tool": "nope", "kind": "timeout"}, {"fs_read"})


def test_validate_fault_rejects_non_json_tool() -> None:
    with pytest.raises(Exception) as excinfo:
        validate_fault({"tool": "calc", "kind": "malformed_json"}, {"calc"})
    assert "malformed_json" in str(excinfo.value)


def test_faults_fire_once_then_clear() -> None:
    world = World.from_spec({"faults": [{"tool": "calc", "kind": "timeout", "times": 1}]})
    assert world.next_fault("calc") is not None
    assert world.next_fault("calc") is None


def test_seed_changes_ticks_but_not_results() -> None:
    args = {"expression": "1 + 1"}
    first = World(seed=1).tick("calc", args)
    second = World(seed=2).tick("calc", args)
    assert 1 <= first <= 4
    assert 1 <= second <= 4
    assert World(seed=1).tick("calc", args) == first
"""Grader behaviour: every grader must be able to fail."""

from __future__ import annotations

from agent_bench.grader import GraderRunner, score, summarise
from agent_bench.models import CallRecord, CheckOutcome, Grader
from agent_bench.world import World


def call(tool: str, ok: bool = True, output: str | None = None, error: str | None = None,
         error_kind: str | None = None, fault: bool = False, **args) -> CallRecord:
    return CallRecord(
        index=1,
        tool=tool,
        args=args,
        ok=ok,
        output=output,
        error=error,
        error_kind=error_kind,
        fault_injected=fault,
        ticks=1,
    )


def run(world: World, graders: list[Grader], calls: list[CallRecord], answer: str,
        claimed: list[str] | None = None) -> list[CheckOutcome]:
    return GraderRunner(world).run(graders, calls, answer, claimed or [])


def grader(kind: str, **params) -> Grader:
    return Grader(kind=kind, params=params, weight=1.0, description=f"{kind} check")


def test_answer_contains_any_of() -> None:
    outcomes = run(
        World(), [grader("answer_contains", any=["alpha", "beta"])], [], "the BETA value"
    )
    assert outcomes[0].passed


def test_answer_contains_fails_cleanly() -> None:
    outcomes = run(World(), [grader("answer_contains", text="gamma")], [], "alpha")
    assert not outcomes[0].passed
    assert "gamma" in outcomes[0].detail


def test_answer_excludes_detects_forbidden_text() -> None:
    outcomes = run(
        World(), [grader("answer_excludes", text="production")], [], "deploy to Production"
    )
    assert not outcomes[0].passed


def test_answer_matches_uses_regex() -> None:
    outcomes = run(World(), [grader("answer_matches", pattern=r"\d+ days")], [], "you have 30 days")
    assert outcomes[0].passed


def test_tool_called_respects_url_filter() -> None:
    calls = [call("http_get", url="/a")]
    assert run(World(), [grader("tool_called", tool="http_get", url="/a")], calls, "")[0].passed
    assert not run(World(), [grader("tool_called", tool="http_get", url="/b")], calls, "")[0].passed


def test_tool_not_called_detects_usage() -> None:
    calls = [call("send_email", to="a@b.c")]
    outcomes = run(World(), [grader("tool_not_called", tool="send_email")], calls, "")
    assert not outcomes[0].passed


def test_file_exists_and_contains() -> None:
    world = World.from_spec({})
    world.write_file("/work/out.txt", "mode: safe")
    outcomes = run(
        world,
        [grader("file_exists", path="/work/out.txt"),
         grader("file_contains", path="/work/out.txt", all=["mode: safe"])],
        [],
        "",
    )
    assert all(outcome.passed for outcome in outcomes)


def test_file_absent_passes_when_untouched() -> None:
    assert run(World(), [grader("file_absent", path="/work/out.txt")], [], "")[0].passed


def test_email_sent_checks_recipient_and_subject() -> None:
    world = World.from_spec({"email_policy": {"allow": ["@example.com"]}})
    world.send_email("ada@example.com", "Your refund was approved", "body")
    outcomes = run(
        world,
        [grader("email_sent", to="ada@example.com", subject_contains="refund")],
        [],
        "",
    )
    assert outcomes[0].passed


def test_email_not_sent_fails_when_outbox_used() -> None:
    world = World.from_spec({})
    world.send_email("ada@example.com", "hi", "body")
    assert not run(world, [grader("email_not_sent")], [], "")[0].passed


def test_event_created_checks_title_and_start() -> None:
    world = World.from_spec({})
    world.create_event("standup", "2026-11-24 09:00")
    passed = run(
        world, [grader("event_created", start="2026-11-24 09:00", title_contains="stand")], [], ""
    )[0]
    assert passed.passed


def test_no_side_effects_flags_writes() -> None:
    calls = [call("fs_write", path="/work/a", content="x")]
    assert not run(World(), [grader("no_side_effects")], calls, "")[0].passed


def test_used_tools_reports_missing_tool() -> None:
    calls = [call("fs_read", path="/work/a")]
    outcome = run(World(), [grader("used_tools", tools=["fs_read", "calc"])], calls, "")[0]
    assert not outcome.passed
    assert "calc" in outcome.detail


def test_max_steps_and_max_calls_of_tool() -> None:
    calls = [call("fs_list", path="/work") for _ in range(3)]
    assert not run(World(), [grader("max_steps", limit=2)], calls, "")[0].passed
    over_budget = run(World(), [grader("max_calls_of_tool", tool="fs_list", limit=1)], calls, "")
    assert not over_budget[0].passed


def test_recovered_requires_fault_then_success() -> None:
    calls = [
        call("http_get", ok=False, fault=True, error_kind="http_500", url="/q"),
        call("http_get", ok=True, output='{"rate": "1.08"}', url="/q"),
    ]
    assert run(World(), [grader("recovered", tool="http_get")], calls, "")[0].passed


def test_recovered_fails_without_retry() -> None:
    calls = [call("http_get", ok=False, fault=True, error_kind="http_500", url="/q")]
    assert not run(World(), [grader("recovered", tool="http_get")], calls, "")[0].passed


def test_claimed_tool_honest_detects_fabrication() -> None:
    calls = [call("fs_read", path="/work/a")]
    outcome = run(
        World(), [grader("claimed_tool_honest")], calls, "done", ["fs_read", "send_email"]
    )[0]
    assert not outcome.passed
    assert "send_email" in outcome.detail


def test_numbers_grounded_rejects_invented_numbers() -> None:
    calls = [call("calc", output='{"result": "472.084"}')]
    assert run(World(), [grader("numbers_grounded")], calls, "the result is 472.084")[0].passed
    invented = run(World(), [grader("numbers_grounded")], calls, "the result is 512.9")[0]
    assert not invented.passed


def test_numbers_grounded_treats_trailing_zeros_as_equal() -> None:
    calls = [call("calc", output='{"result": "100"}')]
    assert run(World(), [grader("numbers_grounded")], calls, "total 100.00")[0].passed


def test_answer_max_words_and_not_empty() -> None:
    assert run(World(), [grader("answer_max_words", limit=3)], [], "one two three")[0].passed
    assert not run(World(), [grader("answer_max_words", limit=2)], [], "one two three")[0].passed
    assert not run(World(), [grader("answer_not_empty")], [], "   ")[0].passed


def test_unknown_grader_kind_fails_closed() -> None:
    outcomes = run(World(), [grader("does_not_exist")], [], "")
    assert not outcomes[0].passed
    assert "unsupported" in outcomes[0].detail


def test_score_is_weighted_and_zero_when_ungraded() -> None:
    outcomes = [
        CheckOutcome(kind="a", label="a", passed=True, weight=3.0, detail=""),
        CheckOutcome(kind="b", label="b", passed=False, weight=1.0, detail=""),
    ]
    assert score(outcomes) == 0.75
    assert score([]) == 0.0


def test_summarise_names_failed_checks() -> None:
    outcomes = [CheckOutcome(kind="a", label="reads the file", passed=False, weight=1.0, detail="")]
    assert "reads the file" in summarise(outcomes)
    assert summarise([]) == "all checks passed"
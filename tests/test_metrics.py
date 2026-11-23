"""Wilson intervals, aggregates and leaderboard ordering."""

from __future__ import annotations

import pytest

from agent_bench.metrics import (
    PolicyReport,
    compare_notes,
    leaderboard,
    summarise,
    wilson,
)
from agent_bench.models import CallRecord, CheckOutcome, EpisodeRecord


def episode(task_id: str, category: str, passed: bool, score: float | None = None,
            calls: int = 1, failed: int = 0) -> EpisodeRecord:
    call_records = [
        CallRecord(index=i, tool="fs_read", args={}, ok=i > failed, output="{}",
                   error=None, error_kind=None, fault_injected=False, ticks=1)
        for i in range(1, calls + 1)
    ]
    return EpisodeRecord(
        task_id=task_id,
        category=category,
        repeat=0,
        policy="scripted",
        answer="",
        stop_reason="finish",
        score=score if score is not None else (1.0 if passed else 0.0),
        passed=passed,
        steps=calls,
        calls=call_records,
        checks=[CheckOutcome(kind="k", label="l", passed=passed, weight=1.0, detail="")],
    )


def test_wilson_empty_sample() -> None:
    interval = wilson(0, 0)
    assert (interval.estimate, interval.low, interval.high) == (0.0, 0.0, 0.0)


def test_wilson_all_successes_has_nonzero_width() -> None:
    interval = wilson(12, 12)
    assert interval.estimate == 1.0
    assert interval.high == 1.0
    assert interval.low < 1.0


def test_wilson_bounds_are_sane() -> None:
    interval = wilson(5, 10)
    assert interval.low < 0.5 < interval.high
    assert interval.low >= 0.0 and interval.high <= 1.0


def test_wilson_matches_known_value() -> None:
    interval = wilson(8, 10)
    assert round(interval.low, 3) == 0.490
    assert round(interval.high, 3) == 0.943


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson(5, 3)


def test_summarise_folds_episodes_into_rates() -> None:
    episodes = [
        episode("a", "tool-selection", True),
        episode("b", "tool-selection", False),
        episode("c", "error-recovery", True, calls=4, failed=2),
    ]
    report = summarise("scripted", "core", 1, episodes)
    assert report.episode_count == 3
    assert report.task_count == 3
    assert report.success_rate == pytest.approx(2 / 3)
    assert report.efficiency == pytest.approx(2.0)
    assert report.wasted_rate == pytest.approx(2 / 6)
    assert report.categories["tool-selection"].episodes == 2


def test_report_handles_zero_calls() -> None:
    report = summarise("p", "s", 1, [])
    assert report.wasted_rate == 0.0
    assert report.recovery_rate == 0.0
    assert report.interval.total == 0


def test_recovery_rate_uses_injected_faults() -> None:
    record = episode("a", "error-recovery", True)
    record.injected_faults = 3
    record.recovered_faults = 2
    report = summarise("p", "s", 1, [record])
    assert report.recovery_rate == pytest.approx(2 / 3)


def test_schema_rate_counts_schema_violations() -> None:
    record = episode("a", "tool-selection", False, calls=4)
    record.calls[0] = CallRecord(index=1, tool="fs_read", args={}, ok=False, output=None,
                                 error="bad", error_kind="schema_violation",
                                 fault_injected=False, ticks=1)
    report = summarise("p", "s", 1, [record])
    assert report.schema_violations == 1
    assert report.schema_rate == pytest.approx(0.25)


def test_leaderboard_orders_by_success_then_efficiency() -> None:
    strong = summarise("strong", "s", 1, [episode("a", "tool-selection", True, calls=2)])
    weak = summarise("weak", "s", 1, [episode("a", "tool-selection", False)])
    tidy = summarise("tidy", "s", 1, [episode("a", "tool-selection", True, calls=1)])
    order = [report.policy for report in leaderboard([weak, strong, tidy])]
    assert order[0] == "tidy"
    assert order[-1] == "weak"


def test_compare_notes_reports_separation() -> None:
    good = summarise("scripted", "s", 1, [episode("a", "tool-selection", True)])
    bad = summarise("scripted-naive", "s", 1, [episode("a", "tool-selection", False)])
    notes = compare_notes(good, bad)
    assert any("tool-selection" in note for note in notes)


def test_policy_report_to_dict_is_json_safe() -> None:
    import json

    report = summarise("p", "s", 2, [episode("a", "tool-selection", True)])
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["success_rate"] == 1.0
    assert payload["repeats"] == 2
    assert isinstance(PolicyReport, type)

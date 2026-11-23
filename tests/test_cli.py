"""End-to-end CLI behaviour, exit codes and rendered artefacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent_bench.cli import main
from agent_bench.metrics import summarise
from agent_bench.models import EpisodeRecord
from agent_bench.reports import build_bundle, markdown_leaderboard


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "results"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "agent-bench 1.0.0" in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage: agent-bench" in out
    assert "new-task" in out


def test_list_shows_tasks_and_tools(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "17 tasks" in out
    assert "read-the-token" in out
    assert "11 mock tools" in out


def test_list_json_is_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["suites"]) == 2
    assert "fs_read" in payload["tools"]


def test_list_tools_shows_fault_kinds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--tools"]) == 0
    out = capsys.readouterr().out
    assert "malformed_json" in out
    assert "finish" in out


def test_run_writes_three_artefacts(out_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--suite", "core", "--repeat", "1", "--out", str(out_dir),
                 "--persona", "careful"])
    assert code == 0
    assert (out_dir / "results.json").exists()
    assert (out_dir / "leaderboard.md").exists()
    report = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "<svg" in report
    assert "agent-bench report" in report


def test_run_results_json_shape(out_dir: Path) -> None:
    main(["run", "--suite", "core", "--repeat", "1", "--out", str(out_dir), "--persona", "careful"])
    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["tool"] == "agent-bench"
    assert payload["tasks"] == 12
    policy = payload["policies"][0]
    assert 0.0 <= policy["success_rate"] <= 1.0
    assert "success_ci" in policy
    assert len(payload["episodes"]) == 12


def test_run_is_byte_identical_across_invocations(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    main(["run", "--suite", "core", "--repeat", "2", "--out", str(first), "--persona", "careful"])
    main(["run", "--suite", "core", "--repeat", "2", "--out", str(second), "--persona", "careful"])
    left = json.loads((first / "results.json").read_text(encoding="utf-8"))
    right = json.loads((second / "results.json").read_text(encoding="utf-8"))
    left.pop("started_at")
    right.pop("started_at")
    assert left == right


def test_run_with_both_personas_reports_separation(
    out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--suite", "core", "--repeat", "1", "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "separation:" in out
    assert "scripted-naive" in out


def test_fail_under_gate_exits_one(out_dir: Path) -> None:
    code = main(["run", "--suite", "core", "--repeat", "1", "--out", str(out_dir),
                 "--persona", "naive", "--fail-under", "0.9"])
    assert code == 1


def test_fail_under_passes_for_reference_policy(out_dir: Path) -> None:
    code = main(["run", "--suite", "core", "--repeat", "1", "--out", str(out_dir),
                 "--persona", "careful", "--fail-under", "0.5"])
    assert code == 0


def test_fail_under_out_of_range_is_usage_error(out_dir: Path) -> None:
    assert main(["run", "--suite", "core", "--out", str(out_dir), "--fail-under", "1.5"]) == 2


def test_repeat_out_of_range_is_usage_error(out_dir: Path) -> None:
    assert main(["run", "--suite", "core", "--out", str(out_dir), "--repeat", "0"]) == 2


def test_unknown_suite_is_a_friendly_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", "--suite", "does-not-exist", "--out", str(tmp_path / "r")])
    assert code == 2
    err = capsys.readouterr().err
    assert "error [suite]" in err
    assert "Traceback" not in err


def test_unknown_adapter_is_a_friendly_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", "--adapter", "bogus", "--out", str(tmp_path / "r")])
    assert code == 2
    assert "error [adapter]" in capsys.readouterr().err


def test_json_only_prints_a_document(out_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--suite", "hard", "--repeat", "1", "--out", str(out_dir),
                 "--persona", "careful", "--json-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suite"] == "hard"


def test_quiet_suppresses_the_per_policy_blocks(out_dir: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    main(["run", "--suite", "hard", "--repeat", "1", "--out", str(out_dir),
          "--persona", "careful", "--quiet"])
    out = capsys.readouterr().out
    assert "leaderboard" in out
    assert "episodes" not in out.split("leaderboard")[0]


def test_report_renders_markdown_from_results(tmp_path: Path,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r"
    main(["run", "--suite", "core", "--repeat", "1", "--out", str(out), "--persona", "careful"])
    target = tmp_path / "leaderboard_render.md"
    assert main(["report", "--results", str(out / "results.json"), "--out", str(target)]) == 0
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# agent-bench leaderboard")
    assert "| policy |" in text


def test_report_missing_results_is_usage_error(tmp_path: Path) -> None:
    assert main(["report", "--results", str(tmp_path / "nope.json")]) == 2


def test_report_rejects_foreign_json(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")
    assert main(["report", "--results", str(path)]) == 2


def test_new_task_writes_a_runnable_suite(tmp_path: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "custom.json"
    assert main(["new-task", "--id", "demo-task", "--out", str(target)]) == 0
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["tasks"][0]["id"] == "demo-task"
    assert main(["run", "--suite", str(target), "--out", str(tmp_path / "out"),
                 "--persona", "careful"]) == 0


def test_new_task_refuses_to_clobber_without_force(tmp_path: Path) -> None:
    target = tmp_path / "custom.json"
    target.write_text("{}", encoding="utf-8")
    assert main(["new-task", "--id", "x", "--out", str(target)]) == 2
    assert target.read_text(encoding="utf-8") == "{}"


def test_new_task_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "custom.json"
    target.write_text("{}", encoding="utf-8")
    assert main(["new-task", "--id", "x", "--out", str(target), "--force"]) == 0
    assert "tasks" in target.read_text(encoding="utf-8")


def test_new_task_rejects_unsafe_id(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["new-task", "--id", "bad id/../"]) == 2


def test_markdown_leaderboard_contains_the_formula_hint() -> None:
    report = summarise("p", "s", 1, [
        EpisodeRecord(task_id="t", category="tool-selection", repeat=0, policy="p",
                      answer="", stop_reason="finish", score=1.0, passed=True, steps=1)
    ])
    bundle = build_bundle("s", "suites/s.json", 1, 0, [report], 1, "2026-01-01T00:00:00Z")
    text = markdown_leaderboard(bundle)
    assert "Wilson interval" in text
    assert "100.0%" in text


def test_help_mentions_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert re.search(r"exit codes: 0 ok, 1 quality gate failed, 2 usage error", out)
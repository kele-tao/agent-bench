"""Command line interface.

Exit codes are part of the contract:

* 0 - the run completed and met every requested threshold
* 1 - the run completed but the quality gate (--fail-under) was not met
* 2 - usage error, bad suite, bad adapter or unreadable input
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .adapters import build_adapter
from .errors import AgentBenchError, UsageError
from .metrics import PolicyReport, summarise
from .models import CATEGORIES
from .reports import build_bundle, markdown_leaderboard, write_html, write_json
from .runner import run_episode
from .suite import (
    discover_suites,
    resolve_suite,
    scaffold_suite,
    validate_tool_catalogue,
)
from .tools import MOCK_TOOLS, TOOL_SCHEMAS
from .world import FAULT_KINDS

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"
MAGENTA = "\x1b[35m"
RESET = "\x1b[0m"

DEFAULT_OUT = "results"


# ---------------------------------------------------------------------------
# colour helpers (honour NO_COLOR and non-tty stdout, but FORCE_COLOR wins)
# ---------------------------------------------------------------------------


def colour_enabled(stream: object = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    target = stream if stream is not None else sys.stdout
    return bool(getattr(target, "isatty", lambda: False)())


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap(text, BOLD)

    def dim(self, text: str) -> str:
        return self._wrap(text, DIM)

    def red(self, text: str) -> str:
        return self._wrap(text, RED)

    def green(self, text: str) -> str:
        return self._wrap(text, GREEN)

    def yellow(self, text: str) -> str:
        return self._wrap(text, YELLOW)

    def cyan(self, text: str) -> str:
        return self._wrap(text, CYAN)

    def magenta(self, text: str) -> str:
        return self._wrap(text, MAGENTA)

    def rate(self, value: float) -> str:
        text = f"{value * 100:5.1f}%"
        if value >= 0.9:
            return self.green(text)
        if value >= 0.6:
            return self.yellow(text)
        return self.red(text)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-bench",
        description=(
            "Offline, deterministic benchmark for tool-using LLM agents: 17 tasks, "
            "12 mock tools, injected failures, Wilson intervals and a leaderboard."
        ),
        epilog=(
            "examples:\n"
            "  agent-bench list\n"
            "  agent-bench run --adapter scripted --suite core --repeat 3\n"
            "  agent-bench report --results results/results.json --out report.html\n"
            "  agent-bench new-task --id my-task --category tool-selection\n"
            "\nexit codes: 0 ok, 1 quality gate failed, 2 usage error"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"agent-bench {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    listing = sub.add_parser("list", help="list bundled suites, tasks and mock tools")
    listing.add_argument("--json", action="store_true", help="emit machine readable JSON")
    listing.add_argument("--tools", action="store_true", help="only show the tool catalogue")

    run = sub.add_parser("run", help="run a suite against one or more adapters")
    run.add_argument(
        "--adapter",
        default="scripted",
        help="scripted (reference + naive personas), naive, or openai "
        "(requires AGENT_BENCH_API_KEY); default: scripted",
    )
    run.add_argument("--suite", default="core", help="bundled suite name or path to a suite JSON")
    run.add_argument("--repeat", type=int, default=1, help="repeat every task N times (default 1)")
    run.add_argument("--seed", type=int, default=0, help="seed recorded with the run (default 0)")
    run.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: results)")
    run.add_argument(
        "--persona",
        choices=("both", "careful", "hasty", "naive"),
        default="both",
        help="which scripted personas to run (default: both = careful + hasty + naive)",
    )
    run.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="RATE",
        help="exit 1 when the leading policy's success rate is below RATE (0..1)",
    )
    run.add_argument("--quiet", action="store_true", help="only print the leaderboard")
    run.add_argument(
        "--json-only",
        action="store_true",
        help="print the results JSON to stdout instead of tables",
    )

    report = sub.add_parser("report", help="re-render a report from a results.json")
    report.add_argument(
        "--results", default=f"{DEFAULT_OUT}/results.json", help="path to results.json"
    )
    report.add_argument("--out", default="report.html", help="output .html or .md path")
    report.add_argument("--markdown", action="store_true", help="render Markdown instead of HTML")

    new_task = sub.add_parser("new-task", help="scaffold a new task or suite from the template")
    new_task.add_argument("--id", required=True, help="task id, e.g. find-the-invoice")
    new_task.add_argument(
        "--category", default="tool-selection", choices=CATEGORIES, help="task category"
    )
    new_task.add_argument("--out", default=None, help="file to write (default: stdout)")
    new_task.add_argument("--force", action="store_true", help="overwrite an existing file")
    return parser


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    style = Style(colour_enabled())
    if args.tools:
        return _print_tools(style, as_json=args.json)
    suites = discover_suites()
    if args.json:
        payload = {
            "version": __version__,
            "suites": [
                {
                    "name": suite["name"],
                    "path": suite["path"],
                    "description": suite["description"],
                    "tasks": [task.to_dict() for task in suite["tasks"]],
                }
                for suite in suites
            ],
            "tools": sorted(TOOL_SCHEMAS),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(style.bold(f"agent-bench {__version__}") + style.dim("  offline agent evaluation"))
    print()
    total = 0
    for suite in suites:
        total += len(suite["tasks"])
        print(
            style.cyan(suite["name"].ljust(8))
            + style.dim(suite["path"].ljust(20))
            + f"{len(suite['tasks'])} tasks  "
            + suite["description"]
        )
        for task in suite["tasks"]:
            print(
                "    "
                + task.id.ljust(28)
                + style.magenta(task.category.ljust(24))
                + style.dim(f"{len(task.graders)} grader(s), {task.max_steps} steps")
            )
    print()
    print(f"{total} tasks across {len(suites)} suites, {len(MOCK_TOOLS)} mock tools.")
    print(style.dim("next: agent-bench run --adapter scripted --suite core --repeat 3"))
    return 0


def _print_tools(style: Style, as_json: bool = False) -> int:
    if as_json:
        print(json.dumps({name: TOOL_SCHEMAS[name] for name in sorted(TOOL_SCHEMAS)}, indent=2))
        return 0
    print(style.bold(f"{len(TOOL_SCHEMAS)} tools") + style.dim("  (strict JSON schemas)"))
    header = "  " + "tool".ljust(14) + "required".ljust(28) + "notes"
    print(style.dim(header))
    for name in sorted(TOOL_SCHEMAS):
        schema = TOOL_SCHEMAS[name]
        required = ", ".join(schema.get("required", [])) or "-"
        note = "terminal action" if name == "finish" else "mock tool"
        print("  " + name.ljust(14) + required.ljust(28) + style.dim(note))
    print()
    print(style.dim("fault kinds: " + ", ".join(FAULT_KINDS)))
    return 0


def _adapter_names(kind: str, persona: str) -> list[str]:
    if kind == "scripted":
        if persona == "careful":
            return ["scripted"]
        if persona == "hasty":
            return ["hasty"]
        if persona == "naive":
            return ["naive"]
        return ["scripted", "hasty", "naive"]
    if kind == "openai":
        if persona != "both":
            raise UsageError("--persona only applies to --adapter scripted")
        return ["openai"]
    return [kind]


def cmd_run(args: argparse.Namespace) -> int:
    style = Style(colour_enabled())
    validate_tool_catalogue()
    if args.repeat < 1 or args.repeat > 50:
        raise UsageError(f"--repeat must be between 1 and 50 (got {args.repeat})")
    if args.fail_under is not None and not 0.0 <= args.fail_under <= 1.0:
        raise UsageError(f"--fail-under must be between 0 and 1 (got {args.fail_under})")

    suite_name, tasks = resolve_suite(args.suite)
    adapters = [build_adapter(name) for name in _adapter_names(args.adapter, args.persona)]

    reports: list[PolicyReport] = []
    for adapter in adapters:
        episodes = []
        for repeat in range(args.repeat):
            for task in tasks:
                episodes.append(run_episode(task, adapter, repeat=repeat))
        reports.append(summarise(adapter.name, suite_name, args.repeat, episodes))

    bundle = build_bundle(
        suite=suite_name,
        suite_path=args.suite,
        repeats=args.repeat,
        seed=args.seed,
        reports=reports,
        tasks_total=len(tasks),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(bundle, str(out_dir / "results.json"))
    (out_dir / "leaderboard.md").write_text(
        markdown_leaderboard(bundle) + "\n", encoding="utf-8", newline="\n"
    )
    write_html(bundle, str(out_dir / "report.html"))

    if args.json_only:
        print(json.dumps(bundle.to_dict(), indent=2))
    else:
        _render_run(style, bundle, tasks, quiet=args.quiet)
        print(style.dim(f"wrote {out_dir / 'results.json'}, {out_dir / 'leaderboard.md'}, "
                        f"{out_dir / 'report.html'}"))

    leading = bundle.ordered()[0]
    if args.fail_under is not None and leading.success_rate < args.fail_under:
        if not args.json_only:
            print(
                style.red(
                    f"quality gate failed: {leading.policy} scored "
                    f"{leading.success_rate * 100:.1f}% < {args.fail_under * 100:.1f}%"
                )
            )
        return 1
    return 0


def _render_run(style: Style, bundle, tasks, quiet: bool = False) -> None:
    header = (
        style.bold(f"agent-bench {__version__}")
        + "  suite "
        + style.cyan(bundle.suite)
        + f" ({len(tasks)} tasks)  x{bundle.repeats}"
        + style.dim(f"  seed {bundle.seed}")
    )
    print(header)
    print()

    if not quiet:
        for report in bundle.ordered():
            print(style.bold(report.policy) + style.dim(f"  {report.episode_count} episodes"))
            print(
                "  task success  "
                + style.rate(report.success_rate)
                + style.dim(
                    f"  95% CI [{report.interval.low * 100:.1f}, "
                    f"{report.interval.high * 100:.1f}]"
                )
                + style.dim("  mean score ")
                + f"{report.mean_score * 100:.1f}%"
            )
            recovery = (
                f"{report.recovery_rate * 100:.0f}%" if report.injected_faults else "n/a"
            )
            print(
                "  efficiency    "
                + f"{report.efficiency:.2f} calls/task"
                + style.dim("   wasted ")
                + f"{report.wasted_rate * 100:.1f}%"
                + style.dim("   recovery ")
                + recovery
                + style.dim("   schema ")
                + f"{report.schema_rate * 100:.1f}%"
            )
            failed = [episode for episode in report.episodes if not episode.passed]
            seen_ids: set[str] = set()
            unique_failed = [
                episode
                for episode in failed
                if not (episode.task_id in seen_ids or seen_ids.add(episode.task_id))
            ]
            for episode in unique_failed[:3]:
                labels = "; ".join(
                    check.label for check in episode.checks if not check.passed
                )
                labels = "; ".join(labels.split("; ")[:2])
                print(style.dim("  missed ") + style.red(episode.task_id.ljust(28)) + labels)
            if len(unique_failed) > 3:
                print(style.dim(f"  ... {len(unique_failed) - 3} more failing tasks"))
            print()

    print(style.bold("leaderboard"))
    print(style.dim("  #  policy            success              mean   calls  wasted  recovery"))
    for index, report in enumerate(bundle.ordered(), start=1):
        recovery = (
            f"{report.recovery_rate * 100:.0f}%" if report.injected_faults else "n/a"
        )
        print(
            "  "
            + str(index).ljust(3)
            + report.policy.ljust(18)
            + style.rate(report.success_rate)
            + style.dim(f" [{report.interval.low * 100:4.1f},{report.interval.high * 100:5.1f}]")
            + f"  {report.mean_score * 100:5.1f}%"
            + f"  {report.efficiency:5.2f}"
            + f"  {report.wasted_rate * 100:5.1f}%"
            + f"    {recovery}"
        )
    print()

    if len(bundle.reports) > 1:
        ordered = bundle.ordered()
        best, worst = ordered[0], ordered[-1]
        gap = (best.success_rate - worst.success_rate) * 100
        verdict = style.green("discriminates") if gap >= 20 else style.yellow("weak separation")
        print(
            "separation: "
            + verdict
            + f"  {best.policy} {best.success_rate * 100:.0f}% vs "
            + f"{worst.policy} {worst.success_rate * 100:.0f}% "
            + style.dim(f"({gap:+.0f} points)")
        )
    print(style.dim("exit codes: 0 ok, 1 quality gate failed, 2 usage error"))



def cmd_report(args: argparse.Namespace) -> int:
    style = Style(colour_enabled())
    path = Path(args.results)
    if not path.exists():
        raise UsageError(
            f"results file not found: {path}",
            hint="run 'agent-bench run' first, or pass --results",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise UsageError(f"{path} is not valid JSON: {exc}") from exc
    if payload.get("tool") != "agent-bench" or "policies" not in payload:
        raise UsageError(f"{path} does not look like an agent-bench results file")

    bundle = _bundle_from_payload(payload)
    out = Path(args.out)
    if args.markdown or out.suffix.lower() == ".md":
        out.write_text(markdown_leaderboard(bundle) + "\n", encoding="utf-8", newline="\n")
    else:
        write_html(bundle, str(out))
    print(style.green("wrote ") + str(out))
    return 0


def _bundle_from_payload(payload: dict) -> object:
    """Rebuild a RunBundle from a results.json without re-running anything."""

    from .metrics import CategoryStats, Interval, PolicyReport
    from .models import EpisodeRecord
    from .reports import RunBundle

    reports: list[PolicyReport] = []
    episodes = payload.get("episodes", [])
    for entry in payload.get("policies", []):
        report = PolicyReport(
            policy=entry["policy"],
            suite=entry["suite"],
            repeats=entry["repeats"],
        )
        report.episodes = [
            EpisodeRecord(
                task_id=item["task_id"],
                category=item["category"],
                repeat=item["repeat"],
                policy=item["policy"],
                answer=item["answer"],
                stop_reason=item["stop_reason"],
                score=item["score"],
                passed=item["passed"],
                steps=item["steps"],
                calls=[],
                checks=[],
                recovered_faults=item.get("recovered_faults", 0),
                injected_faults=item.get("injected_faults", 0),
                claimed_tools=item.get("claimed_tools", []),
            )
            for item in episodes
            if item["policy"] == entry["policy"]
        ]
        from .models import CheckOutcome

        raw_episodes = [item for item in episodes if item["policy"] == entry["policy"]]
        for episode, raw in zip(report.episodes, raw_episodes, strict=False):
            episode.checks = [
                CheckOutcome(
                    kind=check["kind"],
                    label=check["label"],
                    passed=check["passed"],
                    weight=check["weight"],
                    detail=check["detail"],
                )
                for check in raw.get("checks", [])
            ]
        report.categories = {}
        for stats in entry.get("categories", []):
            interval = stats.get("ci") or {}
            report.categories[stats["category"]] = CategoryStats(
                category=stats["category"],
                episodes=stats["episodes"],
                passed=stats["passed"],
                calls=stats.get("calls", 0),
                failed_calls=stats.get("wasted_calls", 0),
                interval=Interval(
                    successes=interval.get("successes", 0),
                    total=interval.get("total", 0),
                    low=interval.get("low", 0.0),
                    high=interval.get("high", 0.0),
                    estimate=interval.get("estimate", 0.0),
                ),
            )
        reports.append(report)

    return RunBundle(
        suite=payload.get("suite", "unknown"),
        suite_path=payload.get("suite_path", ""),
        repeats=payload.get("repeats", 1),
        seed=payload.get("seed", 0),
        started_at=payload.get("started_at", ""),
        reports=reports,
        tasks_total=payload.get("tasks", 0),
    )


def cmd_new_task(args: argparse.Namespace) -> int:
    if not args.id or not all(ch.isalnum() or ch in "-_" for ch in args.id):
        raise UsageError(
            f"task id {args.id!r} must be alphanumeric, '-' or '_'",
            hint="for example: --id find-the-invoice",
        )
    suite = scaffold_suite(args.id, args.category)
    text = json.dumps(suite, indent=2, ensure_ascii=False) + "\n"
    if args.out is None:
        sys.stdout.write(text)
        return 0
    out = Path(args.out)
    if out.exists() and not args.force:
        raise UsageError(
            f"{out} already exists",
            hint="pass --force to overwrite it",
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    style = Style(colour_enabled())
    print(style.green("scaffolded ") + str(out))
    print(style.dim("next: edit intent.steps + graders, then run "
                    f"--suite {out.as_posix()} --repeat 1"))
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    handlers = {
        "list": cmd_list,
        "run": cmd_run,
        "report": cmd_report,
        "new-task": cmd_new_task,
    }
    try:
        return handlers[args.command](args)
    except AgentBenchError as exc:
        print(exc.render(), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0
    except Exception as exc:  # pragma: no cover - last resort, never a bare traceback
        print(
            f"error [internal]: {type(exc).__name__}: {exc}\n"
            "  hint: this is a bug; please report it with the command you ran",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
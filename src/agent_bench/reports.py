"""Render results as a JSON document, a Markdown leaderboard and a standalone HTML report.

The HTML report is a single file with no external assets: the charts are inline
SVG generated from the same numbers that go into results.json. That keeps the
report archivable and makes the screenshot trivially reproducible.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import __version__
from .metrics import PolicyReport, compare_notes, leaderboard
from .models import EpisodeRecord

BACKTICK = chr(96)


@dataclass
class RunBundle:
    """The full artefact set of one run: reports plus the raw episodes."""

    suite: str
    suite_path: str
    repeats: int
    seed: int
    started_at: str
    reports: list[PolicyReport]
    tasks_total: int

    def ordered(self) -> list[PolicyReport]:
        return leaderboard(self.reports)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "tool": "agent-bench",
            "version": __version__,
            "suite": self.suite,
            "suite_path": self.suite_path,
            "repeats": self.repeats,
            "seed": self.seed,
            "started_at": self.started_at,
            "tasks": self.tasks_total,
            "policies": [report.to_dict() for report in self.ordered()],
            "episodes": [
                episode.to_dict() for report in self.ordered() for episode in report.episodes
            ],
        }


def build_bundle(suite: str, suite_path: str, repeats: int, seed: int,
                 reports: Iterable[PolicyReport], tasks_total: int,
                 started_at: str | None = None) -> RunBundle:
    return RunBundle(
        suite=suite,
        suite_path=suite_path,
        repeats=repeats,
        seed=seed,
        started_at=started_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        reports=list(reports),
        tasks_total=tasks_total,
    )


def write_json(bundle: RunBundle, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(bundle.to_dict(), handle, indent=2, sort_keys=False)
        handle.write("\n")


def markdown_leaderboard(bundle: RunBundle) -> str:
    """A GitHub-flavoured leaderboard table plus per-category detail."""

    lines: list[str] = []
    lines.append("# agent-bench leaderboard - suite " + BACKTICK + bundle.suite + BACKTICK)
    lines.append("")
    lines.append(
        f"{bundle.tasks_total} tasks x {bundle.repeats} repeat(s), "
        f"agent-bench {__version__}, run at {bundle.started_at}."
    )
    lines.append("")
    lines.append(
        "| # | policy | task success | mean score | calls/episode | wasted | recovery | schema |"
    )
    lines.append(
        "|---|--------|--------------|------------|---------------|--------|----------|--------|"
    )
    for index, report in enumerate(bundle.ordered(), start=1):
        lines.append(
            "| {rank} | {policy} | {ci} | {score} | {eff} | {waste} | "
            "{recovery} | {schema} |".format(
                rank=index,
                policy=report.policy,
                ci=report.interval.render(),
                score=f"{report.mean_score * 100:.1f}%",
                eff=f"{report.efficiency:.2f}",
                waste=f"{report.wasted_rate * 100:.0f}%",
                recovery=(
                    f"{report.recovery_rate * 100:.0f}%"
                    if report.injected_faults
                    else "n/a"
                ),
                schema=f"{report.schema_rate * 100:.0f}%",
            )
        )
    lines.append("")
    lines.append(
        "Task success counts a task as passed only when every grader passes; the bracket "
        "is the 95% Wilson interval over episodes."
    )
    lines.append("")

    categories: list[str] = []
    for report in bundle.ordered():
        for stats in report.categories.values():
            if stats.episodes and stats.category not in categories:
                categories.append(stats.category)
    lines.append("## Per-category success rate")
    lines.append("")
    header = "| category | " + " | ".join(report.policy for report in bundle.ordered()) + " |"
    lines.append(header)
    lines.append("|---" * (len(bundle.ordered()) + 1) + "|")
    for category in categories:
        cells = []
        for report in bundle.ordered():
            stats = report.categories.get(category)
            if not stats or not stats.episodes:
                cells.append("-")
            else:
                cells.append(f"{stats.success_rate * 100:.0f}% ({stats.interval.render()})")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)

def _bar_svg(value: float, width: int = 200, height: int = 12,
             colour: str = "#2dd4bf") -> str:
    filled = max(1, int(round(value * width))) if value > 0 else 0
    label = f"{value * 100:.1f} percent"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{label}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="6" fill="#1e293b"/>'
        f'<rect x="0" y="0" width="{filled}" height="{height}" rx="6" fill="{colour}"/>'
        "</svg>"
    )


def _ci_svg(estimate: float, low: float, high: float, width: int = 200,
            height: int = 12) -> str:
    left = int(round(low * width))
    right = max(left + 2, int(round(high * width)))
    dot = int(round(estimate * width))
    label = (
        f"point estimate {estimate * 100:.1f} percent, interval "
        f"{low * 100:.1f} to {high * 100:.1f} percent"
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{label}">'
        f'<line x1="0" y1="{height // 2}" x2="{width}" y2="{height // 2}" '
        f'stroke="#243244" stroke-width="2"/>'
        f'<rect x="{left}" y="3" width="{right - left}" height="{height - 6}" rx="2" '
        f'fill="#0ea5e9" opacity="0.55"/>'
        f'<circle cx="{dot}" cy="{height // 2}" r="3.5" fill="#f8fafc"/>'
        "</svg>"
    )


HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>agent-bench report</title>
<style>
body{margin:0;background:#0b1120;color:#e2e8f0;
     font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 28px 64px}
h1{font-size:30px;margin:0 0 6px}
h2{font-size:19px;margin:38px 0 12px;color:#f1f5f9}
.sub{color:#94a3b8;margin:0 0 22px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #1f2b3d;vertical-align:middle}
th{color:#94a3b8;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
td.num{font-variant-numeric:tabular-nums}
.card{background:#111a2b;border:1px solid #1f2b3d;border-radius:12px;
      padding:16px 20px;margin:14px 0}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;
      background:#13293d;color:#7dd3fc;border:1px solid #1e3a5f;margin-right:6px}
.fail{color:#fca5a5}
.pass{color:#86efac}
code{background:#16202f;padding:1px 5px;border-radius:4px;font-size:13px}
.muted{color:#64748b;font-size:13px}
</style></head><body><div class="wrap">
"""


def html_report(bundle: RunBundle) -> str:
    """A self-contained HTML report with inline SVG bars and no external assets."""

    order = bundle.ordered()
    parts: list[str] = [HTML_HEAD]
    parts.append("<h1>agent-bench report</h1>")
    parts.append(
        f"<p class='sub'>suite <code>{html.escape(bundle.suite)}</code> &middot; "
        f"{bundle.tasks_total} tasks &middot; {bundle.repeats} repeat(s) &middot; "
        f"seed {bundle.seed} &middot; {html.escape(bundle.started_at)}</p>"
    )
    parts.append(
        "<p class='muted'>Task success counts a task as passed only when every grader "
        "passes. The bar is the point estimate; the blue whisker is the 95% Wilson "
        "interval over episodes.</p>"
    )
    parts.append(
        "<h2>Leaderboard</h2><table><tr><th>#</th><th>policy</th>"
        "<th>success (95% CI)</th><th>mean score</th><th>calls/ep</th>"
        "<th>wasted</th><th>recovery</th><th>schema</th></tr>"
    )
    for index, report in enumerate(order, start=1):
        recovery = (
            f"{report.recovery_rate * 100:.0f}%" if report.injected_faults else "n/a"
        )
        parts.append(
            f"<tr><td class='num'>{index}</td>"
            f"<td><b>{html.escape(report.policy)}</b></td>"
            f"<td>{_ci_svg(report.interval.estimate, report.interval.low, report.interval.high)}"
            f"<br><span class='num'>{report.interval.render()}</span></td>"
            f"<td class='num'>{report.mean_score * 100:.1f}%<br>"
            f"{_bar_svg(report.mean_score, colour='#38bdf8')}</td>"
            f"<td class='num'>{report.efficiency:.2f}</td>"
            f"<td class='num'>{report.wasted_rate * 100:.0f}%</td>"
            f"<td class='num'>{recovery}</td>"
            f"<td class='num'>{report.schema_rate * 100:.0f}%</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h2>Per category</h2><table><tr><th>category</th>")
    for report in order:
        parts.append(f"<th>{html.escape(report.policy)}</th>")
    parts.append("</tr>")
    categories: list[str] = []
    for report in order:
        for stats in report.categories.values():
            if stats.episodes and stats.category not in categories:
                categories.append(stats.category)
    for category in categories:
        parts.append(f"<tr><td>{html.escape(category)}</td>")
        for report in order:
            stats = report.categories.get(category)
            if not stats or not stats.episodes:
                parts.append("<td class='muted'>-</td>")
                continue
            parts.append(
                f"<td>{_bar_svg(stats.success_rate)} "
                f"<span class='num'>{stats.success_rate * 100:.0f}%</span></td>"
            )
        parts.append("</tr>")
    parts.append("</table>")

    for report in order:
        parts.append(f"<h2>{html.escape(report.policy)} - episode detail</h2>")
        failed = [episode for episode in report.episodes if not episode.passed]
        parts.append(
            f"<div class='card'><span class='pill'>success {report.interval.render()}</span>"
            f"<span class='pill'>{report.episode_count} episodes</span>"
            f"<span class='pill'>{report.total_calls} tool calls</span>"
            f"<span class='pill'>{report.injected_faults} injected faults, "
            f"{report.recovered_faults} recovered</span></div>"
        )
        if not failed:
            parts.append("<p class='pass'>Every episode passed.</p>")
            continue
        parts.append(
            "<table><tr><th>task</th><th>category</th><th>score</th>"
            "<th>stop</th><th>calls</th><th>failed checks</th></tr>"
        )
        for episode in failed[:12]:
            labels = "; ".join(
                html.escape(check.label) for check in episode.checks if not check.passed
            )
            parts.append(
                f"<tr><td>{html.escape(episode.task_id)}</td>"
                f"<td class='muted'>{html.escape(episode.category)}</td>"
                f"<td class='num'>{episode.score * 100:.0f}%</td>"
                f"<td class='muted'>{html.escape(episode.stop_reason)}</td>"
                f"<td class='num'>{episode.steps}</td>"
                f"<td class='fail'>{labels}</td></tr>"
            )
        parts.append("</table>")
        if len(failed) > 12:
            parts.append(
                f"<p class='muted'>{len(failed) - 12} more failing episodes are in "
                "results.json.</p>"
            )

    by_name = {report.policy: report for report in order}
    if "scripted" in by_name and "scripted-naive" in by_name:
        notes = compare_notes(by_name["scripted"], by_name["scripted-naive"])
        parts.append(
            "<h2>Separation check</h2><div class='card'><p class='muted'>The careful "
            "reference policy and the naive baseline run against identical worlds. A "
            "suite that cannot separate them is not measuring anything.</p><ul>"
        )
        for note in notes:
            parts.append(f"<li>{html.escape(note)}</li>")
        parts.append("</ul></div>")

    parts.append("</div></body></html>")
    return "\n".join(parts)


def write_html(bundle: RunBundle, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(html_report(bundle))
        handle.write("\n")


def episode_lines(episodes: Iterable[EpisodeRecord]) -> list[dict[str, Any]]:
    """Flatten episodes for JSONL consumers."""

    return [episode.to_dict() for episode in episodes]
#!/usr/bin/env python
"""Generate the benchmark numbers quoted in the README.

    python benchmarks/run.py

Writes `benchmarks/results.json` and `benchmarks/BENCHMARK.md`. Every number in the
README's performance section comes from re-running this script, so a reviewer
can reproduce it on their own laptop. Timings are wall clock around the episode
loop only (not process start-up) and are reported as the median of five runs.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_bench import __version__  # noqa: E402
from agent_bench.adapters import ScriptedAdapter  # noqa: E402
from agent_bench.metrics import summarise  # noqa: E402
from agent_bench.runner import run_episode  # noqa: E402
from agent_bench.suite import load_bundled_suite  # noqa: E402

SUITES = ("core", "hard")
PERSONAS = ("careful", "hasty", "naive")
TIMING_RUNS = 5


def run_suite(suite_name: str, persona: str, repeats: int = 1):
    adapter = ScriptedAdapter(persona=persona)
    _, tasks = load_bundled_suite(suite_name)
    episodes = [
        run_episode(task, adapter, repeat=repeat)
        for repeat in range(repeats)
        for task in tasks
    ]
    return summarise(adapter.name, suite_name, repeats, episodes), len(tasks)


def time_suite(suite_name: str, persona: str) -> float:
    adapter = ScriptedAdapter(persona=persona)
    _, tasks = load_bundled_suite(suite_name)
    samples = []
    for _ in range(TIMING_RUNS):
        start = time.perf_counter()
        for task in tasks:
            run_episode(task, adapter)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def large_synthetic_suite(size: int = 200):
    """Build a synthetic suite with N tasks to measure per-task scaling."""

    from agent_bench.suite import parse_suite

    task = {
        "id": "synthetic",
        "category": "long-context-retrieval",
        "title": "Retrieve a value from a large document set",
        "instruction": "Search the documentation and report the value of metric-{i}.",
        "intent": {
            "steps": [{"id": "s1", "tool": "search_docs",
                       "args": {"query": "metric value", "top_k": 3}}],
            "answer": "The value is {s1.hits[0].text}",
        },
        "world": {
            "docs": [
                {"title": f"doc-{doc}", "text": f"metric-{doc} value is {doc * 7}"}
                for doc in range(50)
            ]
        },
        "graders": [
            {"kind": "tool_called", "weight": 1.0, "description": "searched the docs",
             "params": {"tool": "search_docs"}},
        ],
    }
    tasks = []
    for index in range(size):
        clone = json.loads(json.dumps(task))
        clone["id"] = f"synthetic-{index:03d}"
        tasks.append(clone)
    _, parsed = parse_suite({"name": "synthetic", "tasks": tasks})
    return parsed


def main() -> int:
    report = {"version": __version__, "suites": {}, "timing": {}}

    for suite_name in SUITES:
        entries = {}
        for persona in PERSONAS:
            summary, task_count = run_suite(suite_name, persona)
            entries[summary.policy] = {
                **summary.to_dict(),
                "task_count": task_count,
            }
        report["suites"][suite_name] = entries

    for suite_name in SUITES:
        report["timing"][suite_name] = {
            "median_ms": round(time_suite(suite_name, "careful"), 2),
            "runs": TIMING_RUNS,
        }

    synthetic = large_synthetic_suite(200)
    adapter = ScriptedAdapter(persona="careful")
    start = time.perf_counter()
    for task in synthetic:
        run_episode(task, adapter)
    elapsed = (time.perf_counter() - start) * 1000.0
    report["timing"]["synthetic_200_tasks"] = {
        "median_ms": round(elapsed, 2),
        "runs": 1,
        "tasks": len(synthetic),
    }

    total_tests = count_tests()
    report["tests"] = total_tests

    out_dir = ROOT / "benchmarks"
    (out_dir / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / "BENCHMARK.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print(f"wrote {out_dir / 'results.json'}")
    print(f"wrote {out_dir / 'BENCHMARK.md'}")
    return 0


def count_tests() -> int:
    """Ask pytest how many tests it collects; fall back to counting defs."""

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        match = re.search(r"(\d+) tests? collected", completed.stdout)
        if match:
            return int(match.group(1))
        counted = 0
        for line in completed.stdout.splitlines():
            match = re.match(r"^tests/.*:\s*(\d+)$", line.strip())
            if match:
                counted += int(match.group(1))
        if counted:
            return counted
    except (OSError, subprocess.SubprocessError):
        pass
    total = 0
    for path in pathlib.Path(ROOT / "tests").glob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        total += text.count("\ndef test_") + (1 if text.startswith("def test_") else 0)
    return total


def render_markdown(report: dict) -> str:
    lines = ["# agent-bench benchmark results", ""]
    lines.append(
        f"Generated by `python benchmarks/run.py` with agent-bench "
        f"{report['version']}. Do not edit by hand."
    )
    lines.append("")
    lines.append("## Policy separation")
    lines.append("")
    for suite_name, entries in report["suites"].items():
        lines.append(f"### suite `{suite_name}`")
        lines.append("")
        lines.append(
            "| policy | task success | 95% CI | mean score | calls/task | wasted | recovery |"
        )
        lines.append(
            "|--------|--------------|--------|------------|------------|--------|----------|"
        )
        for policy in sorted(entries, key=lambda name: -entries[name]["success_rate"]):
            entry = entries[policy]
            ci = entry["success_ci"]
            recovery = (
                f"{entry['recovery_rate'] * 100:.0f}%" if entry["injected_faults"] else "n/a"
            )
            lines.append(
                f"| {policy} | {entry['success_rate'] * 100:.1f}% "
                f"| [{ci['low'] * 100:.1f}, {ci['high'] * 100:.1f}] "
                f"| {entry['mean_score'] * 100:.1f}% "
                f"| {entry['efficiency']:.2f} "
                f"| {entry['wasted_rate'] * 100:.0f}% | {recovery} |"
            )
        lines.append("")
    lines.append("## Episode-loop timing")
    lines.append("")
    lines.append("| workload | median wall clock |")
    lines.append("|----------|-------------------|")
    for name, entry in report["timing"].items():
        unit = f"{entry['median_ms']:.1f} ms"
        if "tasks" in entry:
            unit += f" ({entry['tasks']} tasks)"
        lines.append(f"| {name} | {unit} |")
    lines.append("")
    lines.append(f"Test suite: **{report['tests']} tests**.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
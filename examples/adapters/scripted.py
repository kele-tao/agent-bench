#!/usr/bin/env python
"""Two adapters, one benchmark: a reference agent and a deliberately broken one.

Run it:

    python examples/adapters/scripted.py

It executes every task in both bundled suites against

* the *careful* reference policy, which uses the declared plan, retries after
  injected faults and copies numbers out of tool output; and
* the *naive* policy, which skips lookups, gives up on the first fault and
  quotes figures it never observed.

The point of the example is the gap. If your suite cannot separate these two,
it is measuring nothing, and the script exits non-zero so CI notices.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from agent_bench.adapters import ScriptedAdapter  # noqa: E402
from agent_bench.metrics import summarise  # noqa: E402
from agent_bench.runner import run_episode  # noqa: E402
from agent_bench.suite import load_bundled_suite  # noqa: E402


def evaluate(persona: str, repeats: int = 1):
    """Run every bundled suite with one persona and return one report per suite."""

    adapter = ScriptedAdapter(persona=persona)
    reports = []
    for suite_name in ("core", "hard"):
        _, tasks = load_bundled_suite(suite_name)
        episodes = [
            run_episode(task, adapter, repeat=repeat)
            for repeat in range(repeats)
            for task in tasks
        ]
        reports.append(summarise(adapter.name, suite_name, repeats, episodes))
    return reports


def main() -> int:
    careful = {report.suite: report for report in evaluate("careful")}
    naive = {report.suite: report for report in evaluate("naive")}

    print("suite   careful   naive    delta     careful mean score")
    for suite in ("core", "hard"):
        good = careful[suite]
        bad = naive[suite]
        delta = (good.success_rate - bad.success_rate) * 100
        print(
            f"{suite:<7} {good.success_rate * 100:6.1f}%  {bad.success_rate * 100:6.1f}%"
            f"  {delta:+6.0f} pts   {good.mean_score * 100:5.1f}%"
        )

    gap = sum(r.success_rate for r in careful.values()) - sum(
        r.success_rate for r in naive.values()
    )
    print()
    if gap <= 0:
        print("FAIL: the suite does not separate a careful agent from a broken one.")
        return 1
    print(f"OK: the benchmark separates the two policies by {gap * 100 / 2:.0f} points on average.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

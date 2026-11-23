# Roadmap

What is in 1.0.0, what was deliberately cut, and what comes next.

## Shipped in 1.0.0

- 17 tasks (12 `core` + 5 `hard`) across 8 categories.
- 12 tools with strict JSON schemas and 18 grader kinds.
- Scripted adapters with three personas: `careful`, `hasty`, `naive`.
- Injected `timeout` / `http_500` / `malformed_json` faults, armed per task.
- Metrics with Wilson 95% intervals, per-category breakdown, recovery and
  schema-violation rates.
- `results.json`, `leaderboard.md` and a self-contained `report.html` with inline
  SVG charts.
- `list` / `run` / `report` / `new-task` commands, exit codes 0/1/2.

## Cut from 1.0.0 (and why)

| idea | why it was cut |
|---|---|
| Streaming / latency metrics | A mock world has no meaningful latency. Reporting one would be a made-up number. `ticks` is recorded for cost reporting only and is not scored. |
| Multi-agent topologies | The adapter protocol is single-agent. Adding a topology layer would double the surface for a feature no bundled task needs. |
| Web UI / hosted leaderboard | Out of scope for an offline tool; `report.html` covers the visual need with zero infrastructure. |
| Token accounting for the OpenAI adapter | Depends on a specific provider's usage payload; deferred until there is a real model run to attribute cost to. |
| A `jsonschema` dependency | The needed subset is ~90 lines and the error messages are better for LLM retries. |

## Next

1. **Replay adapter.** Re-score a recorded model transcript offline. Highest
   value feature: it makes real-model runs reproducible without an API key.
2. **Power analysis.** Given a suite size, report the minimum detectable
   difference between two policies, and warn when a suite is too small to
   separate them.
3. **Committed baseline files.** `--baseline results/baseline.json` fails the run
   when a policy regresses against a reference score, not only a threshold.
4. **More ambiguity tasks**, including negative cases where calling `ask_user`
   is itself the failure because the instruction is complete.
5. **A cost axis.** Token accounting for the OpenAI adapter next to tool-call
   efficiency, so "same score, half the calls" is visible.

## Non-goals

* Replacing a real model evaluation. agent-bench scores policy and harness
  behaviour deterministically; it does not claim a model quality ranking.
* Running untrusted code. Tools are mocks, by design.

# Changelog

All notable changes to agent-bench are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-23

### Added

- 17 deterministic tasks across 8 categories: `suites/core.json` (12) and
  `suites/hard.json` (5).
- 12 mock tools with strict JSON schemas, a seeded in-memory filesystem, mock
  HTTP APIs and an exact decimal calculator.
- Fault injection: timeouts, HTTP 500 responses and malformed JSON payloads,
  configured per task so recovery behaviour is measurable.
- `ScriptedAdapter` with two personas (careful reference policy, naive
  baseline) for offline CI runs, plus an optional OpenAI-compatible adapter.
- Metrics with Wilson 95% confidence intervals, per-category breakdown, wasted
  calls, schema violations, recovery rate and steps-to-solution.
- Anti-hallucination checks: claimed tool calls must have happened and factual
  tokens in the answer must be grounded in tool output.
- Markdown leaderboard and a self-contained HTML report with inline SVG charts.
- `agent-bench list`, `run`, `report` and `new-task` commands.

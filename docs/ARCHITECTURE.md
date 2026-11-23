# Architecture

agent-bench is four layers, each replaceable on its own.

`@BT@`text
 suites/*.json            declarative: task, world, faults, graders, intent
      |
   suite.py               parse + validate (fails closed, before any episode)
      |
   runner.py  <--->  adapters.py        episode loop <-> policy
      |                   |
   tools.py  <--->  world.py            mock tool layer <-> scripted state
      |
   grader.py              deterministic checks over calls, side effects, answer
      |
   metrics.py             Wilson intervals, efficiency, recovery, leaderboard
      |
   reports.py             results.json, leaderboard.md, report.html
`@BT@`

## Module map

| module | responsibility | notable invariant |
|---|---|---|
| `suite.py` | load and validate suites, scaffold new tasks | a suite that cannot run is rejected before any episode starts |
| `models.py` | dataclasses shared by every layer | no side effects; import-safe from anywhere |
| `world.py` | per-episode state: files, HTTP routes, tables, docs, email policy, faults | rebuilt per episode, never shared |
| `tools.py` | 12 tool schemas, validators, handlers, fault injection, calculator | schema violations never execute a handler |
| `runner.py` | the episode loop and its four stop reasons | one episode = one fresh world |
| `adapters.py` | scripted personas, template resolution, OpenAI bridge | network only behind `AGENT_BENCH_API_KEY` |
| `grader.py` | 18 deterministic grader kinds | every grader can fail; unknown kinds fail closed |
| `metrics.py` | aggregates and Wilson intervals | rates always carry an interval |
| `reports.py` | JSON, Markdown, self-contained HTML | no external assets, no JS |
| `cli.py` | argument parsing, rendering, exit codes | 0 ok, 1 gate failed, 2 usage error |

## The episode loop

`@BT@`text
world   = World.from_spec(task.world, seed=task.seed, faults=task.faults)
adapter.reset(task)
repeat up to task.max_steps:
    action = adapter.next_action(observation)
    if action is finish:  -> grade, stop_reason = "finish"
    if action is tool:    -> call_tool(world, tool, args)  (validates, injects fault, records)
    if the same tool+args is seen 3 times -> stop_reason = "repeated_call"
    if a sandbox escape is attempted        -> stop_reason = "sandbox_escape"
otherwise                               -> stop_reason = "max_steps"
`@BT@`

The loop never raises for policy mistakes. It records them. An adapter that
throws is an adapter bug, and becomes an `AdapterError` with a friendly message.

## Why faults are armed in the world, not in the adapter

If the adapter knew a fault was coming, its retry policy would be scripted by
the benchmark rather than by the policy author. Arming the fault in the world
means the adapter only observes the tool result, exactly as a real agent would.
The cost of recovery - an extra call, an extra step - is therefore measured
identically across every policy.

## Why the world is rebuilt per episode

`--repeat 3` must produce three independent samples. A shared world would make
the second and third repeats trivially different from the first (files already
written, emails already queued), and `results.json` would stop being
reproducible. The cost is rebuilding a handful of dicts; the timestamp is the
only non-deterministic field in the output.

## Determinism

* No wall-clock reads inside the graded path (only `started_at` in the bundle).
* No unseeded randomness; the `ticks` pseudo-latency is a SHA-256 of
  `(seed, tool, args)`.
* Sorted iteration everywhere results are serialised (`sort_keys=True`,
  sorted category order, sorted tool catalogue).
* The test suite asserts byte equality of two full runs apart from `started_at`.

## Extending

1. **New tool** - add a schema to `TOOL_SCHEMAS`, a handler to `HANDLERS`, a world
   method, and tests. Then add it to `JSON_TOOLS` if its payload is JSON.
2. **New grader** - add `_check_<kind>` to `GraderRunner`; it is registered
   automatically through `STRICT_GRADERS`. Add the required params to
   `GRADER_PARAMS` in `suite.py`.
3. **New adapter** - implement `reset(task)` and `next_action(observation)`;
   nothing else needs to change.

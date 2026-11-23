# Contributing

Thanks for taking the time to improve agent-bench.

## Development setup

```bash
git clone https://github.com/kele-tao/agent-bench
cd agent-bench
python -m pip install -e ".[dev]"
pytest -q
ruff check .
python -m agent_bench run --adapter scripted --suite core
```

## Ground rules

1. **Determinism is a feature.** Anything that changes `results/results.json`
   between two identical runs is a bug. No wall clocks, no unseeded randomness,
   no network access in the test suite.
2. **No invented numbers.** If you touch the harness, re-run
   `python benchmarks/run.py` and commit the regenerated files.
3. **Every new tool needs a schema and tests.** Add it to `src/agent_bench/tools.py`,
   cover the happy path plus at least one error path in `tests/test_tools.py`.
4. **Every new task needs a grader that can fail.** A task that every policy
   passes measures nothing. Add it to `suites/core.json` or `suites/hard.json`
   and confirm the naive baseline still loses points.
5. **Style:** `ruff check .` must be clean; line length is 100; public
   functions have type hints.

## Adding a task

```bash
agent-bench new-task --id my-task --category tool-selection --out suites/my.json
agent-bench run --suite suites/my.json --repeat 1
```

Edit the generated JSON: set a precise instruction, the structured `intent`
(consumed by the scripted reference policy), optional `faults`, and at least one
grader. A task without graders is rejected by the loader.

## Pull requests

Keep pull requests focused. Describe the behaviour change, paste the command you
ran, and mention any metric that moved. CI runs ruff plus pytest on Python
3.10-3.12 and a CLI smoke test.

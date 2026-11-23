# Security policy

agent-bench executes **only** built-in mock tools. It never runs shell commands,
never touches the real filesystem outside the in-memory sandbox under `/work`,
and makes no network requests unless you explicitly opt in by running the
OpenAI-compatible adapter.

## Reporting a vulnerability

If you find a way to escape the mock world (for example a path traversal that
writes outside the in-memory sandbox) or to leak the OpenAI API key into logs,
please open a private security advisory on GitHub or email the maintainer listed
on the profile. Do not open a public issue with a working exploit.

You can expect an acknowledgement within 72 hours and a fix or mitigation plan
within 14 days.

## Supported versions

| Version | Supported |
| ------- | --------- |
| 1.x     | yes       |

## API keys

The optional OpenAI-compatible adapter reads `AGENT_BENCH_API_KEY` from the
environment. The value is never written to results files and never printed.
`agent-bench run --adapter openai` is opt-in; the default adapter is offline.

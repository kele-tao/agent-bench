"""Policy adapters.

An adapter is anything that can look at the conversation so far and emit the
next action. Three are shipped:

* ScriptedAdapter with persona="careful" -- the reference policy. It executes
  the task's declared plan through the real tool layer, retries after an
  injected fault and copies its final numbers out of tool output. It is a fixed
  program, not a model: its score is an upper bound that the task design can
  reach, and it is what CI asserts against.
* ScriptedAdapter with persona="naive" -- a deliberately sloppy policy that
  skips lookups, gives up on the first fault and quotes numbers it never
  observed. If a task suite cannot separate the two, the suite is broken.
* OpenAICompatibleAdapter -- opt-in bridge to any /chat/completions endpoint.
  Never used in tests, never used implicitly, needs an env var.

The protocol is intentionally tiny so a real agent framework can be wrapped in
about twenty lines: reset(task) once per episode, then next_action(observation)
returning either a tool call or a final answer.
"""

from __future__ import annotations

import json
import os
import re
import string
import urllib.error
import urllib.request
from typing import Any, Protocol

from .errors import AdapterError
from .models import Task, Turn

PERSONAS = ("careful", "hasty", "naive")


class Observation:
    """What the adapter knows at the start of a step."""

    def __init__(self, task: Task, step: int, last: Any, history: list[Any]) -> None:
        self.task = task
        self.step = step
        self.last = last
        self.history = history

    def transcript(self) -> list[Turn]:
        turns = [Turn(role="user", content=self.task.instruction)]
        for index, result in enumerate(self.history, start=1):
            if result.ok:
                turns.append(Turn(role="tool", name=result.tool, content=result.output or ""))
            else:
                turns.append(
                    Turn(
                        role="tool",
                        name=result.tool,
                        content=f"ERROR ({result.error_kind}): {result.error}",
                    )
                )
            turns.append(Turn(role="assistant", content=f"step {index}: {result.tool}"))
        return turns

    def history_text(self) -> str:
        """Compact transcript used by the optional OpenAI adapter."""

        return "\n".join(turn.render() for turn in self.transcript())


class Adapter(Protocol):
    name: str

    def reset(self, task: Task) -> None: ...

    def next_action(self, observation: Observation) -> dict[str, Any]: ...


class TemplateError(Exception):
    """Raised when an answer template cannot be resolved from tool output."""


def resolve(template: str, results: dict[str, Any]) -> str:
    """Fill {step.field[0].field} placeholders from parsed tool output.

    Values are rendered from the tool payload, so an answer such as
    {total.result} can never introduce floating point noise.
    """

    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise TemplateError(f"malformed answer template: {exc}") from exc

    out: list[str] = []
    for literal, field_name, format_spec, conversion in parsed:
        out.append(literal)
        if field_name is None:
            continue
        if conversion is not None or format_spec:
            raise TemplateError("answer templates do not support conversions or format specs")
        out.append(_render(_lookup(field_name, results)))
    return "".join(out)


def _lookup(field_name: str, results: dict[str, Any]) -> Any:
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", field_name)
    if not parts:
        raise TemplateError(f"empty placeholder {field_name!r}")
    head = parts[0]
    if head not in results:
        raise TemplateError(f"placeholder references unknown step {head!r}")
    value: Any = results[head]
    for part in parts[1:]:
        if part.startswith("["):
            index = int(part[1:-1])
            if not isinstance(value, list) or index >= len(value):
                raise TemplateError(f"index {index} is out of range for {field_name!r}")
            value = value[index]
        else:
            if not isinstance(value, dict) or part not in value:
                raise TemplateError(f"field {part!r} is missing for {field_name!r}")
            value = value[part]
    return value


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, sort_keys=True)


class ScriptedAdapter:
    """A deterministic policy driven by the task's declared intent block.

    intent.steps is the correct plan. intent.answer is either a literal string
    or a template over step outputs. intent.naive holds the deliberate mistakes
    of the sloppy persona, so both policies run against exactly the same world.
    """

    def __init__(self, persona: str = "careful", max_retries: int = 2) -> None:
        if persona not in PERSONAS:
            raise AdapterError(
                f"unknown scripted persona {persona!r}",
                hint="valid personas: " + ", ".join(PERSONAS),
            )
        self.persona = persona
        self.name = "scripted" if persona == "careful" else f"scripted-{persona}"
        self.max_retries = max_retries
        self._task: Task | None = None
        self._steps: list[dict[str, Any]] = []
        self._index = 0
        self._retries = 0
        self._results: dict[str, Any] = {}
        self._naive: dict[str, Any] = {}
        self._finished = False

    def reset(self, task: Task) -> None:
        intent = task.intent or {}
        steps = intent.get("steps")
        if not isinstance(steps, list) or not steps:
            raise AdapterError(
                f"task {task.id!r} has no intent.steps, so the scripted adapter cannot plan",
                hint="every task needs at least one step for the scripted adapter",
            )
        self._task = task
        self._steps = [dict(step) for step in steps]
        self._index = 0
        self._retries = 0
        self._results = {}
        self._finished = False
        naive = intent.get("naive") or {}
        self._naive = {
            "skip": list(naive.get("skip", [])),
            "answer": str(naive.get("answer", "I could not complete the task.")),
            "fabricate": list(naive.get("fabricate", [])),
            "claim": list(naive.get("claim", [])),
        }

    def next_action(self, observation: Observation) -> dict[str, Any]:
        if self.persona == "naive":
            return self._next_naive(observation)
        if self.persona == "hasty":
            return self._next_hasty(observation)
        return self._next_careful(observation)

    def _next_hasty(self, observation: Observation) -> dict[str, Any]:
        """Full plan, but gives up the moment a tool misbehaves."""

        last = observation.last
        if last is not None and not last.ok:
            self._finished = True
            return self._finish(
                "The tool failed, so I stopped: " + str(last.error),
                [call.tool for call in observation.history],
            )
        if last is not None and last.ok:
            self._record(last)
        if self._index >= len(self._steps):
            self._finished = True
            return self._finish(self._answer(), [])
        step = self._steps[self._index]
        return {"type": "tool", "tool": step["tool"], "args": dict(step.get("args", {}))}

    # -- personas -------------------------------------------------------------
    def _next_careful(self, observation: Observation) -> dict[str, Any]:
        last = observation.last
        if last is not None and not last.ok:
            if last.fault_injected and self._retries < self.max_retries:
                self._retries += 1
                step = self._steps[self._index]
                return {
                    "type": "tool",
                    "tool": step["tool"],
                    "args": dict(step.get("args", {})),
                }
            self._finished = True
            return self._finish("I could not retrieve the data: " + str(last.error), [])
        if last is not None and last.ok:
            self._record(last)
        if self._index >= len(self._steps):
            self._finished = True
            return self._finish(self._answer(), [])
        step = self._steps[self._index]
        return {"type": "tool", "tool": step["tool"], "args": dict(step.get("args", {}))}

    def _next_naive(self, observation: Observation) -> dict[str, Any]:
        if self._finished:
            return self._finish(self._naive["answer"], self._naive["claim"])
        last = observation.last
        if last is not None and last.fault_injected:
            self._finished = True
            return self._finish(self._naive["answer"], self._naive["claim"])
        if last is not None and last.ok:
            self._record(last)
        while self._index < len(self._steps):
            step = self._steps[self._index]
            key = str(step.get("id", f"s{self._index}"))
            if key in self._naive["skip"]:
                self._index += 1
                continue
            return {"type": "tool", "tool": step["tool"], "args": dict(step.get("args", {}))}
        self._finished = True
        return self._finish(self._naive["answer"], self._naive["claim"])

    def _record(self, last: Any) -> None:
        step = self._steps[self._index]
        key = str(step.get("id", f"s{self._index}"))
        self._results[key] = last.payload()
        self._index += 1
        self._retries = 0

    def _answer(self) -> str:
        assert self._task is not None
        template = str(self._task.intent.get("answer", ""))
        if not template:
            return "Task complete."
        try:
            return resolve(template, self._results)
        except TemplateError as exc:
            return f"Task could not be summarised from tool output ({exc})."

    def _finish(self, answer: str, claimed: list[str]) -> dict[str, Any]:
        text = answer
        if self.persona == "naive":
            for extra in self._naive.get("fabricate", []):
                text += " " + str(extra)
        return {"type": "finish", "answer": text, "claimed_tools": [str(c) for c in claimed]}


def _normalise_payload(payload: Any) -> Any:
    """Turn {"result": "12"} into {"result": "12"} and leave structures alone."""

    return payload


class OpenAICompatibleAdapter:
    """Optional bridge to any OpenAI-compatible chat completions endpoint.

    Disabled by default. Enabled only when AGENT_BENCH_API_KEY is set and the
    caller passes --adapter openai. The key is read once and never logged or
    written to results files.
    """

    TOOL_PROTOCOL = (
        "You can call tools. Reply with exactly one JSON object per turn:\n"
        '{"tool": "<name>", "args": {...}} to call a tool, or\n'
        '{"answer": "<final answer>", "claimed_tools": ["<tool>", ...]} when done.\n'
        "Never state a number you have not seen in a tool result."
    )

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 700,
    ) -> None:
        from .tools import TOOL_SCHEMAS

        self.api_key = api_key or os.environ.get("AGENT_BENCH_API_KEY", "")
        if not self.api_key:
            raise AdapterError(
                "the openai adapter needs AGENT_BENCH_API_KEY in the environment",
                hint="export AGENT_BENCH_API_KEY=... or use --adapter scripted for offline runs",
            )
        self.model = model or os.environ.get("AGENT_BENCH_MODEL", "gpt-4o-mini")
        self.base_url = (
            base_url or os.environ.get("AGENT_BENCH_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.name = "openai:" + self.model
        self._schema_text = json.dumps(TOOL_SCHEMAS, sort_keys=True)
        self._task: Task | None = None

    def reset(self, task: Task) -> None:
        self._task = task

    def next_action(self, observation: Observation) -> dict[str, Any]:
        system = (
            "You are being evaluated by agent-bench. Use the tools to gather facts.\n"
            + self.TOOL_PROTOCOL
            + "\nTool schemas:\n"
            + self._schema_text
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": observation.history_text()},
        ]
        return self._parse(self._chat(messages))

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise AdapterError(f"could not reach {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise AdapterError("endpoint returned a non-JSON body") from exc
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError("endpoint response did not contain a chat completion") from exc

    def _parse(self, reply: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", reply, re.S)
        if not match:
            return {"type": "finish", "answer": reply.strip(), "claimed_tools": []}
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return {"type": "finish", "answer": reply.strip(), "claimed_tools": []}
        if "answer" in data:
            claimed = data.get("claimed_tools") or []
            return {
                "type": "finish",
                "answer": str(data["answer"]),
                "claimed_tools": [str(tool) for tool in claimed if isinstance(tool, str)],
            }
        if "tool" in data:
            args = data.get("args")
            return {
                "type": "tool",
                "tool": str(data["tool"]),
                "args": args if args is not None else {},
            }
        return {"type": "finish", "answer": reply.strip(), "claimed_tools": []}


def build_adapter(kind: str, **kwargs: Any) -> Adapter:
    """Construct an adapter by CLI name, with friendly errors."""

    if kind in ("scripted", "reference", "careful"):
        return ScriptedAdapter(persona="careful")
    if kind in ("hasty", "scripted-hasty"):
        return ScriptedAdapter(persona="hasty")
    if kind in ("naive", "baseline", "scripted-naive"):
        return ScriptedAdapter(persona="naive")
    if kind == "openai":
        return OpenAICompatibleAdapter(**kwargs)
    raise AdapterError(
        f"unknown adapter {kind!r}",
        hint="choices: scripted, hasty, naive, openai",
    )


ADAPTER_CHOICES = ("scripted", "hasty", "naive", "openai")
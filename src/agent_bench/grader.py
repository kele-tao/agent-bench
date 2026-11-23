"""Deterministic graders.

Every grader answers one question about an episode and returns a pass/fail plus
a human readable detail string. The task score is the weighted share of grader
weight that passed, so a task with one heavy grader and one light grader
partially credits a policy that only satisfies the heavy one.

The anti-hallucination graders are first class citizens: a policy that claims to
have called a tool it never called, or that quotes numbers absent from every
tool output, loses that grader's weight.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import CallRecord, CheckOutcome, Grader
from .world import World, format_decimal

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _calls(calls: list[CallRecord]) -> list[CallRecord]:
    return calls


def _successful_numbers(calls: list[CallRecord]) -> set[str]:
    """Every numeric token that appeared in a successful tool output."""

    values: set[str] = set()
    for call in calls:
        if not call.ok or not call.output:
            continue
        for token in NUMBER_RE.findall(call.output):
            values.add(token)
            try:
                values.add(format_decimal(__import__("decimal").Decimal(token)))
            except Exception:
                continue
    return values


def _normalise_number(token: str) -> str:
    try:
        return format_decimal(__import__("decimal").Decimal(token))
    except Exception:
        return token


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


class GraderRunner:
    """Evaluate a task's graders against one finished episode."""

    def __init__(self, world: World) -> None:
        self.world = world

    def run(self, graders: list[Grader], calls: list[CallRecord],
            answer: str, claimed_tools: list[str]) -> list[CheckOutcome]:
        outcomes: list[CheckOutcome] = []
        for grader in graders:
            passed, detail = self._check(grader, calls, answer, claimed_tools)
            outcomes.append(
                CheckOutcome(
                    kind=grader.kind,
                    label=grader.description,
                    passed=passed,
                    weight=grader.weight,
                    detail=detail,
                )
            )
        return outcomes

    # -- dispatch ------------------------------------------------------------
    def _check(self, grader: Grader, calls: list[CallRecord], answer: str,
               claimed_tools: list[str]) -> tuple[bool, str]:
        handler = getattr(self, "_check_" + grader.kind.replace("-", "_"), None)
        if handler is None:
            return False, f"unsupported grader kind {grader.kind!r}"
        try:
            return handler(grader.params, calls, answer, claimed_tools)
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"grader misconfigured: {type(exc).__name__}: {exc}"

    # -- graders -------------------------------------------------------------
    def _check_answer_contains(self, params: dict[str, Any], calls: list[CallRecord],
                               answer: str, claimed: list[str]) -> tuple[bool, str]:
        needles = params.get("any") or [params["text"]]
        lowered = answer.lower()
        hits = [needle for needle in needles if needle.lower() in lowered]
        if hits:
            return True, f"answer contains {hits[0]!r}"
        return False, "answer contains none of: " + ", ".join(repr(n) for n in needles)

    def _check_answer_excludes(self, params: dict[str, Any], calls: list[CallRecord],
                               answer: str, claimed: list[str]) -> tuple[bool, str]:
        needles = params.get("any") or [params["text"]]
        lowered = answer.lower()
        hits = [needle for needle in needles if needle.lower() in lowered]
        if hits:
            return False, f"answer contains forbidden text {hits[0]!r}"
        return True, "answer avoids all forbidden text"

    def _check_answer_matches(self, params: dict[str, Any], calls: list[CallRecord],
                              answer: str, claimed: list[str]) -> tuple[bool, str]:
        pattern = params["pattern"]
        flags = re.IGNORECASE if params.get("ignore_case", True) else 0
        if re.search(pattern, answer, flags):
            return True, f"answer matches /{pattern}/"
        return False, f"answer does not match /{pattern}/"

    def _check_tool_called(self, params: dict[str, Any], calls: list[CallRecord],
                           answer: str, claimed: list[str]) -> tuple[bool, str]:
        tool = params["tool"]
        matching = [c for c in calls if c.tool == tool and (c.ok or params.get("allow_failure"))]
        if not matching:
            return False, f"{tool} was never called successfully"
        if "path_suffix" in params:
            suffix = params["path_suffix"]
            args_ok = [c for c in matching
                       if str(c.args.get("path", "")).endswith(suffix)]
            if not args_ok:
                return False, f"{tool} never targeted a path ending in {suffix!r}"
        if "url" in params:
            urls = [c for c in matching if c.args.get("url") == params["url"]]
            if not urls:
                return False, f"{tool} never called {params['url']!r}"
        if "table" in params:
            tables = [c for c in matching if c.args.get("table") == params["table"]]
            if not tables:
                return False, f"{tool} never queried table {params['table']!r}"
        return True, f"{tool} called ({len(matching)}x)"

    def _check_tool_not_called(self, params: dict[str, Any], calls: list[CallRecord],
                               answer: str, claimed: list[str]) -> tuple[bool, str]:
        tool = params["tool"]
        matching = [c for c in calls if c.tool == tool]
        if matching:
            return False, f"{tool} was called {len(matching)}x but must not be called"
        return True, f"{tool} never called"

    def _check_file_exists(self, params: dict[str, Any], calls: list[CallRecord],
                           answer: str, claimed: list[str]) -> tuple[bool, str]:
        from .world import normalise_path

        path = normalise_path(params["path"])
        if path in self.world.files:
            return True, f"{path} exists in the sandbox"
        return False, f"{path} was never written"

    def _check_file_contains(self, params: dict[str, Any], calls: list[CallRecord],
                             answer: str, claimed: list[str]) -> tuple[bool, str]:
        from .world import normalise_path

        path = normalise_path(params["path"])
        body = self.world.files.get(path, "")
        for needle in params.get("all", []):
            if needle.lower() not in body.lower():
                return False, f"{path} is missing {needle!r}"
        return True, f"{path} contains all required text"

    def _check_file_absent(self, params: dict[str, Any], calls: list[CallRecord],
                           answer: str, claimed: list[str]) -> tuple[bool, str]:
        from .world import normalise_path

        path = normalise_path(params["path"])
        if path in self.world.files:
            return False, f"{path} should not have been created"
        return True, f"{path} was not created"

    def _check_email_sent(self, params: dict[str, Any], calls: list[CallRecord],
                          answer: str, claimed: list[str]) -> tuple[bool, str]:
        to = params.get("to", "").lower()
        suffix = params.get("to_suffix")
        matches = [
            message
            for message in self.world.outbox
            if not to
            or message["to"] == to
            or (suffix and message["to"].endswith(suffix))
        ]
        if not matches:
            return False, "no email was queued" + (f" to {to}" if to else "")
        if "subject_contains" in params:
            wanted = params["subject_contains"].lower()
            if not any(wanted in m["subject"].lower() for m in matches):
                return False, f"email subject never contained {params['subject_contains']!r}"
        return True, f"{len(matches)} email(s) queued"

    def _check_email_not_sent(self, params: dict[str, Any], calls: list[CallRecord],
                              answer: str, claimed: list[str]) -> tuple[bool, str]:
        if self.world.outbox:
            return False, f"{len(self.world.outbox)} email(s) were sent but none were allowed"
        return True, "no email was sent"

    def _check_event_created(self, params: dict[str, Any], calls: list[CallRecord],
                             answer: str, claimed: list[str]) -> tuple[bool, str]:
        start = params.get("start")
        if not self.world.events:
            return False, "no calendar event was created"
        if start and not any(e["start"] == start for e in self.world.events):
            return False, f"no event at {start!r}"
        if "title_contains" in params:
            wanted = params["title_contains"].lower()
            if not any(wanted in e["title"].lower() for e in self.world.events):
                return False, f"no event title contained {params['title_contains']!r}"
        return True, f"{len(self.world.events)} event(s) created"

    def _check_no_side_effects(self, params: dict[str, Any], calls: list[CallRecord],
                               answer: str, claimed: list[str]) -> tuple[bool, str]:
        writes = [c for c in calls if c.tool in ("fs_write", "send_email", "create_event")]
        if writes:
            names = ", ".join(sorted({c.tool for c in writes}))
            return False, f"mutating tools were called: {names}"
        return True, "no mutating tool was called"

    def _check_used_tools(self, params: dict[str, Any], calls: list[CallRecord],
                          answer: str, claimed: list[str]) -> tuple[bool, str]:
        wanted = params["tools"]
        used = {c.tool for c in calls if c.ok}
        missing = [tool for tool in wanted if tool not in used]
        if missing:
            return False, "never used: " + ", ".join(missing)
        return True, "used every required tool"

    def _check_max_steps(self, params: dict[str, Any], calls: list[CallRecord],
                         answer: str, claimed: list[str]) -> tuple[bool, str]:
        limit = int(params["limit"])
        if len(calls) <= limit:
            return True, f"{len(calls)} calls <= {limit} allowed"
        return False, f"{len(calls)} calls exceeds the budget of {limit}"

    def _check_max_calls_of_tool(self, params: dict[str, Any], calls: list[CallRecord],
                                 answer: str, claimed: list[str]) -> tuple[bool, str]:
        tool = params["tool"]
        limit = int(params["limit"])
        count = sum(1 for c in calls if c.tool == tool)
        if count <= limit:
            return True, f"{count} call(s) of {tool} within budget {limit}"
        return False, f"{count} calls of {tool} exceeds budget {limit}"

    def _check_recovered(self, params: dict[str, Any], calls: list[CallRecord],
                         answer: str, claimed: list[str]) -> tuple[bool, str]:
        tool = params["tool"]
        faulted = False
        for call in calls:
            if call.fault_injected and call.tool == tool:
                faulted = True
                continue
            if faulted and call.tool == tool and call.ok:
                return True, f"retried {tool} after an injected fault and succeeded"
        if not faulted:
            return False, f"no injected fault was observed for {tool}"
        return False, f"{tool} was never retried successfully after the fault"

    def _check_claimed_tool_honest(self, params: dict[str, Any], calls: list[CallRecord],
                                   answer: str, claimed: list[str]) -> tuple[bool, str]:
        called = {call.tool for call in calls}
        fabricated = sorted({tool for tool in claimed if tool not in called})
        if fabricated:
            return False, "claimed calls to tools that were never invoked: " + ", ".join(fabricated)
        if not claimed:
            return True, "no tool claims were made"
        return True, f"all {len(claimed)} claimed tool call(s) really happened"

    def _check_numbers_grounded(self, params: dict[str, Any], calls: list[CallRecord],
                                answer: str, claimed: list[str]) -> tuple[bool, str]:
        grounded = _successful_numbers(calls)
        grounded |= {_normalise_number(v) for v in grounded}
        if "allow" in params:
            grounded |= {str(v) for v in params["allow"]}
        tokens = NUMBER_RE.findall(answer)
        fabricated = [
            token for token in tokens
            if token not in grounded and _normalise_number(token) not in grounded
        ]
        if fabricated:
            listed = ", ".join(fabricated[:5])
            return False, "answer quotes numbers absent from tool output: " + listed
        return True, f"all {len(tokens)} numeric token(s) traceable to tool output"

    def _check_answer_max_words(self, params: dict[str, Any], calls: list[CallRecord],
                                answer: str, claimed: list[str]) -> tuple[bool, str]:
        limit = int(params["limit"])
        count = _word_count(answer)
        if count <= limit:
            return True, f"{count} words <= {limit}"
        return False, f"{count} words exceeds the {limit} word limit"

    def _check_answer_not_empty(self, params: dict[str, Any], calls: list[CallRecord],
                                answer: str, claimed: list[str]) -> tuple[bool, str]:
        if answer.strip():
            return True, "answer is non-empty"
        return False, "answer was empty"

    def _check_stopped_before_finish(self, params: dict[str, Any], calls: list[CallRecord],
                                     answer: str, claimed: list[str]) -> tuple[bool, str]:
        return False, "reserved grader kind"


GRADER_KINDS = tuple(
    name[len("_check_") :]
    for name in dir(GraderRunner)
    if name.startswith("_check_")
)

STRICT_GRADERS = frozenset(
    kind for kind in GRADER_KINDS if kind != "stopped_before_finish"
)


def score(outcomes: list[CheckOutcome]) -> float:
    """Weighted fraction of grader weight that passed; 0.0 when ungraded."""

    total = sum(outcome.weight for outcome in outcomes)
    if total <= 0:
        return 0.0
    earned = sum(outcome.weight for outcome in outcomes if outcome.passed)
    return earned / total


def summarise(outcomes: list[CheckOutcome]) -> str:
    """One-line summary of failed graders, for the CLI and the reports."""

    failed = [outcome.label for outcome in outcomes if not outcome.passed]
    if not failed:
        return "all checks passed"
    return "failed: " + "; ".join(failed[:3])


def dumps_output(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)
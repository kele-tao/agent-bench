"""The deterministic mock world every episode runs against.

An episode never touches the real filesystem, the real network or the real
clock. Everything an agent can observe or change lives in the World dataclass,
which is rebuilt from the task definition for every single run. That is what
makes `agent-bench run --repeat 3` produce byte-identical results.

Vendored filesystem root: `/work`. Paths are POSIX style regardless of host OS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .errors import SuiteError

ROOT = "/work"

FAULT_KINDS = ("timeout", "http_500", "malformed_json")

# Tools whose payload is JSON and can therefore be corrupted by the
# `malformed_json` fault. Text tools only accept timeout/http_500.
JSON_TOOLS = frozenset({"fs_list", "http_get", "db_query", "search_docs"})


class SandboxError(Exception):
    """Raised when an agent tries to escape the `/work` sandbox."""


class ToolFailure(Exception):
    """Raised by a world method for an expected, recoverable tool failure."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass
class FaultPlan:
    """A scripted fault that is armed for a specific tool."""

    tool: str
    kind: str
    remaining: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "kind": self.kind, "remaining": self.remaining}


@dataclass
class ToolResult:
    """The outcome of one tool call, whether it succeeded or not."""

    tool: str
    args: dict[str, Any]
    ok: bool
    output: str | None = None
    error: str | None = None
    error_kind: str | None = None
    fault_injected: bool = False
    ticks: int = 0
    call_index: int = 0

    def payload(self) -> Any:
        """Parsed JSON payload for JSON tools, or None when unparseable."""
        if not self.ok or self.output is None:
            return None
        try:
            return json.loads(self.output)
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "error_kind": self.error_kind,
            "fault_injected": self.fault_injected,
            "ticks": self.ticks,
            "call_index": self.call_index,
        }


def normalise_path(path: str) -> str:
    """Map any agent supplied path onto a canonical path inside `/work`.

    Rejects Windows drive letters, home-relative paths and any parent segment
    that would climb above the sandbox root.
    """

    if not isinstance(path, str) or not path.strip():
        raise SandboxError("path must be a non-empty string")
    raw = path.replace("\\", "/").strip()
    if raw.startswith("~"):
        raise SandboxError(f"home directory paths are not available: {path!r}")
    if len(raw) >= 2 and raw[1] == ":":
        raise SandboxError(f"absolute host paths are not available: {path!r}")
    raw = raw.lstrip("/")
    if raw == "work":
        raw = ""
    elif raw.startswith("work/"):
        raw = raw[len("work/") :]
    parts: list[str] = []
    for segment in raw.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not parts:
                raise SandboxError(f"path escapes the sandbox root: {path!r}")
            parts.pop()
            continue
        if segment.endswith(" ") or segment.endswith("."):
            raise SandboxError(f"path has a trailing dot or space: {path!r}")
        parts.append(segment)
    return ROOT + ("/" + "/".join(parts) if parts else "")


def format_decimal(value: Decimal) -> str:
    """Render a Decimal without scientific notation or float noise."""

    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


@dataclass
class World:
    """Mutable state for one episode. Rebuild it; never share it."""

    files: dict[str, str] = field(default_factory=dict)
    http_routes: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    docs: list[dict[str, str]] = field(default_factory=list)
    clarifications: list[dict[str, str]] = field(default_factory=list)
    email_policy: dict[str, list[str]] = field(default_factory=dict)
    faults: list[FaultPlan] = field(default_factory=list)
    seed: int = 0

    # -- observed side effects ------------------------------------------------
    outbox: list[dict[str, str]] = field(default_factory=list)
    events: list[dict[str, str]] = field(default_factory=list)
    calls: list[ToolResult] = field(default_factory=list)

    @classmethod
    def from_spec(
        cls, spec: dict[str, Any], seed: int = 0, faults: list[dict[str, Any]] | None = None
    ) -> World:
        world = cls(seed=seed)
        world.files = {
            normalise_path(path): str(body) for path, body in spec.get("files", {}).items()
        }
        world.http_routes = dict(spec.get("http", {}))
        world.tables = {
            str(name): [dict(row) for row in rows] for name, rows in spec.get("tables", {}).items()
        }
        world.docs = [dict(doc) for doc in spec.get("docs", [])]
        world.clarifications = [dict(item) for item in spec.get("clarifications", [])]
        policy = spec.get("email_policy", {})
        world.email_policy = {
            "allow": [str(d).lower() for d in policy.get("allow", [])],
            "deny": [str(d).lower() for d in policy.get("deny", [])],
        }
        for entry in list(spec.get("faults", [])) + list(faults or []):
            world.faults.append(
                FaultPlan(
                    tool=str(entry["tool"]),
                    kind=str(entry["kind"]),
                    remaining=int(entry.get("times", 1)),
                )
            )
        return world

    # -- lifecycle ------------------------------------------------------------
    def next_fault(self, tool: str) -> FaultPlan | None:
        for plan in self.faults:
            if plan.tool == tool and plan.remaining > 0:
                plan.remaining -= 1
                return plan
        return None

    def tick(self, tool: str, args: dict[str, Any]) -> int:
        """Deterministic pseudo-latency in [1, 4]; used for cost reporting only."""

        blob = f"{self.seed}:{tool}:{sorted(args.items())!r}".encode()
        digest = hashlib.sha256(blob).digest()[0]
        return 1 + digest % 4

    def record(self, result: ToolResult) -> ToolResult:
        result.call_index = len(self.calls) + 1
        self.calls.append(result)
        return result

    # -- tools ----------------------------------------------------------------
    def read_file(self, path: str) -> str:
        key = normalise_path(path)
        if key not in self.files:
            raise ToolFailure("file_not_found", f"no such file: {key}")
        return self.files[key]

    def write_file(self, path: str, content: str) -> str:
        key = normalise_path(path)
        if key == ROOT:
            raise ToolFailure("invalid_path", "cannot write to the sandbox root itself")
        self.files[key] = content
        return f"wrote {len(content)} characters to {key}"

    def list_dir(self, path: str) -> list[dict[str, Any]]:
        key = normalise_path(path)
        prefix = key.rstrip("/") + "/"
        if not any(existing.startswith(prefix) for existing in self.files):
            raise ToolFailure("file_not_found", f"no such directory: {key}")
        seen: dict[str, dict[str, Any]] = {}
        for existing, body in sorted(self.files.items()):
            if not existing.startswith(prefix):
                continue
            rest = existing[len(prefix) :]
            name, _, tail = rest.partition("/")
            entry = seen.setdefault(
                name, {"name": name, "path": prefix + name, "type": "file", "size": 0}
            )
            if tail:
                entry["type"] = "directory"
            else:
                entry["size"] = len(body.encode("utf-8"))
        return sorted(seen.values(), key=lambda entry: (entry["type"], entry["name"]))

    def http_get(self, endpoint: str) -> dict[str, Any]:
        route = self.http_routes.get(endpoint)
        if route is None:
            raise ToolFailure("not_found", f"no route registered for GET {endpoint}")
        status = int(route.get("status", 200))
        if status >= 400:
            detail = str(route.get("message", "")).strip()
            message = f"GET {endpoint} returned {status} {detail}".strip()
            raise ToolFailure(f"http_{status}", message)
        return {"status": status, "endpoint": endpoint, "body": route.get("body")}

    def query_table(self, table: str, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if table not in self.tables:
            raise ToolFailure(
                "unknown_table",
                f"unknown table {table!r}; available: " + ", ".join(sorted(self.tables)),
            )
        rows = self.tables[table]
        filters = where or {}
        unknown = sorted(set(filters) - {key for row in rows for key in row})
        if unknown:
            raise ToolFailure(
                "unknown_column",
                f"unknown column(s) {', '.join(unknown)} in table {table!r}",
            )
        matched = [
            dict(row)
            for row in rows
            if all(row.get(key) == value for key, value in filters.items())
        ]
        matched.sort(key=lambda row: tuple(str(row.get(key, "")) for key in sorted(row)))
        return matched

    def search_docs(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        terms = [term for term in query.lower().split() if term]
        if not terms:
            raise ToolFailure("invalid_query", "search query must contain at least one word")
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for doc in self.docs:
            body = doc.get("text", "").lower()
            title = doc.get("title", "").lower()
            score = sum(body.count(term) + 2 * title.count(term) for term in terms)
            if score:
                scored.append((score, doc.get("title", ""), doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"title": doc.get("title", ""), "text": doc.get("text", ""), "score": score}
            for score, _title, doc in scored[:top_k]
        ]

    def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        address = to.strip().lower()
        if "@" not in address or address.startswith("@") or address.endswith("@"):
            raise ToolFailure("invalid_recipient", f"not a valid email address: {to!r}")
        domain = "@" + address.rsplit("@", 1)[1]
        if domain in self.email_policy.get("deny", []):
            raise ToolFailure(
                "policy_denied",
                f"outbound email to {domain} is blocked by the data-egress policy",
            )
        allowed = self.email_policy.get("allow", [])
        if allowed and domain not in allowed:
            raise ToolFailure(
                "policy_denied",
                f"outbound email to {domain} is not on the allow list",
            )
        record = {"to": address, "subject": subject, "body": body}
        self.outbox.append(record)
        return {"queued": True, "to": address, "subject": subject, "outbox_size": len(self.outbox)}

    def create_event(self, title: str, start: str) -> dict[str, Any]:
        for event in self.events:
            if event["start"] == start:
                raise ToolFailure("conflict", f"{start} is already booked for {event['title']!r}")
        record = {"title": title, "start": start}
        self.events.append(record)
        return {"created": True, **record, "event_count": len(self.events)}

    def ask_user(self, question: str) -> dict[str, Any]:
        needle = question.lower()
        best_score = 0
        best_answer = ""
        for item in self.clarifications:
            match = item.get("match", "").lower()
            if match and match in needle:
                score = len(match)
                if score > best_score:
                    best_score = score
                    best_answer = item.get("answer", "")
        if not best_answer:
            return {
                "answer": (
                    "No answer available for that question. "
                    "Proceed with your best judgement."
                ),
                "matched": False,
            }
        return {"answer": best_answer, "matched": True}


def validate_fault(entry: dict[str, Any], tool_names: set[str]) -> None:
    """Reject fault definitions that could never fire."""

    tool = entry.get("tool")
    kind = entry.get("kind")
    if tool not in tool_names:
        raise SuiteError(
            f"fault references unknown tool {tool!r}",
            hint="valid tools: " + ", ".join(sorted(tool_names)),
        )
    if kind not in FAULT_KINDS:
        raise SuiteError(
            f"fault kind {kind!r} is not supported",
            hint="valid kinds: " + ", ".join(FAULT_KINDS),
        )
    if kind == "malformed_json" and tool not in JSON_TOOLS:
        raise SuiteError(
            f"tool {tool!r} does not return JSON, so kind 'malformed_json' cannot apply",
            hint="malformed_json is valid only for: " + ", ".join(sorted(JSON_TOOLS)),
        )
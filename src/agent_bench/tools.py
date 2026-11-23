"""The mock tool layer: strict JSON schemas, deterministic handlers, injected faults.

Design notes
------------
* Schemas are validated by a small hand written JSON Schema subset rather than a
  third party validator, because the whole point of this package is that it is
  installable offline with zero dependencies.
* A call that violates its schema is **rejected before it executes** and is
  counted as a schema violation. Retrying with corrected arguments is allowed
  and is exactly the behaviour the benchmark wants to reward.
* Faults are scripted in the task definition and fire on the Nth call of a tool,
  so a policy that retries once recovers and a policy that gives up does not.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any

from .errors import ToolInputError
from .world import (
    JSON_TOOLS,
    SandboxError,
    ToolFailure,
    ToolResult,
    World,
    format_decimal,
    normalise_path,
)

MAX_EXPRESSION_LENGTH = 200

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "fs_read": {
        "type": "object",
        "properties": {"path": {"type": "string", "minLength": 1}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "fs_write": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    "fs_list": {
        "type": "object",
        "properties": {"path": {"type": "string", "minLength": 1}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "http_get": {
        "type": "object",
        "properties": {"url": {"type": "string", "pattern": "^/[a-z0-9/_-]*$"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    "http_post": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "pattern": "^/[a-z0-9/_-]*$"},
            "body": {"type": "object"},
        },
        "required": ["url", "body"],
        "additionalProperties": False,
    },
    "db_query": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "minLength": 1},
            "where": {"type": "object"},
        },
        "required": ["table"],
        "additionalProperties": False,
    },
    "calc": {
        "type": "object",
        "properties": {"expression": {"type": "string", "minLength": 1}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    "search_docs": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "send_email": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "minLength": 3},
            "subject": {"type": "string", "minLength": 1},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False,
    },
    "create_event": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "start": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}$"},
        },
        "required": ["title", "start"],
        "additionalProperties": False,
    },
    "ask_user": {
        "type": "object",
        "properties": {"question": {"type": "string", "minLength": 1}},
        "required": ["question"],
        "additionalProperties": False,
    },
    "finish": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "claimed_tools": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
}

MOCK_TOOLS = tuple(sorted(name for name in TOOL_SCHEMAS if name != "finish"))

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def validate_args(schema: dict[str, Any], args: Any, path: str = "args") -> list[str]:
    """Validate `args` against a JSON Schema subset; return human readable errors."""

    errors: list[str] = []
    if not isinstance(args, dict):
        return [f"{path}: expected an object, got {type(args).__name__}"]

    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            errors.append(f"{path}.{key}: required argument is missing")
    if schema.get("additionalProperties") is False:
        for key in sorted(args):
            if key not in properties:
                allowed = ", ".join(sorted(properties)) or "(none)"
                errors.append(f"{path}.{key}: unexpected argument; allowed: {allowed}")

    for key, value in args.items():
        rule = properties.get(key)
        if rule is None:
            continue
        errors.extend(_validate_value(rule, value, f"{path}.{key}"))
    return errors


def _validate_value(rule: dict[str, Any], value: Any, path: str) -> list[str]:
    errors: list[str] = []
    expected = rule.get("type")
    if expected:
        allowed = _JSON_TYPES[expected]
        ok = isinstance(value, allowed)
        if expected in ("integer", "number") and isinstance(value, bool):
            ok = False
        if not ok:
            return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if expected == "string":
        if "minLength" in rule and len(value) < rule["minLength"]:
            errors.append(f"{path}: must be at least {rule['minLength']} character(s)")
        if "pattern" in rule and not re.search(rule["pattern"], value):
            errors.append(f"{path}: does not match {rule['pattern']}")
    if expected in ("integer", "number"):
        if "minimum" in rule and value < rule["minimum"]:
            errors.append(f"{path}: must be >= {rule['minimum']}")
        if "maximum" in rule and value > rule["maximum"]:
            errors.append(f"{path}: must be <= {rule['maximum']}")
    if expected == "array" and "items" in rule:
        for index, item in enumerate(value):
            errors.extend(_validate_value(rule["items"], item, f"{path}[{index}]"))
    if expected == "object" and "properties" in rule:
        errors.extend(validate_args(rule, value, path))
    return errors


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

_BIN_OPS: dict[type, Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: (a / b).to_integral_value(rounding="ROUND_FLOOR"),
    ast.Mod: lambda a, b: a % b,
}
_CALC_FUNCS = {"abs": 1, "round": 2, "min": 2, "max": 2}


def _eval_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolFailure("invalid_expression", "only numeric literals are allowed")
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ToolFailure("invalid_expression", "unary operator is not supported")
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base = _eval_node(node.left)
            exponent = _eval_node(node.right)
            if exponent != exponent.to_integral_value() or abs(exponent) > 12:
                raise ToolFailure(
                    "invalid_expression", "exponent must be an integer between -12 and 12"
                )
            return base**int(exponent)
        handler = _BIN_OPS.get(type(node.op))
        if handler is None:
            raise ToolFailure("invalid_expression", "operator is not supported")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolFailure("division_by_zero", "division by zero")
        return handler(left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _CALC_FUNCS:
            raise ToolFailure(
                "invalid_expression",
                "allowed functions: " + ", ".join(sorted(_CALC_FUNCS)),
            )
        name = node.func.id
        if node.keywords:
            raise ToolFailure("invalid_expression", "keyword arguments are not supported")
        args = [_eval_node(arg) for arg in node.args]
        if len(args) != _CALC_FUNCS[name]:
            raise ToolFailure(
                "invalid_expression", f"{name}() takes {_CALC_FUNCS[name]} argument(s)"
            )
        if name == "abs":
            return abs(args[0])
        if name == "min":
            return min(args[0], args[1])
        if name == "max":
            return max(args[0], args[1])
        places = args[1]
        if places != places.to_integral_value() or not 0 <= places <= 8:
            raise ToolFailure("invalid_expression", "round() places must be an integer 0..8")
        return round(args[0], int(places))
    raise ToolFailure("invalid_expression", f"{type(node).__name__} is not allowed in expressions")


def evaluate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, using decimal arithmetic."""

    text = expression.strip()
    if not text:
        raise ToolFailure("invalid_expression", "expression must not be empty")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ToolFailure(
            "invalid_expression", f"expression longer than {MAX_EXPRESSION_LENGTH} characters"
        )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolFailure("invalid_expression", f"could not parse expression: {exc.msg}") from exc
    try:
        return format_decimal(_eval_node(tree))
    except DivisionByZero as exc:
        raise ToolFailure("division_by_zero", "division by zero") from exc
    except InvalidOperation as exc:
        raise ToolFailure("invalid_expression", "arithmetic operation overflowed") from exc


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def _handle_fs_read(world: World, args: dict[str, Any]) -> str:
    path = normalise_path(args["path"])
    body = world.read_file(path)
    return json.dumps({"path": path, "content": body, "bytes": len(body.encode("utf-8"))},
                      sort_keys=True)


def _handle_fs_write(world: World, args: dict[str, Any]) -> str:
    path = normalise_path(args["path"])
    content = args["content"]
    world.write_file(path, content)
    return json.dumps(
        {"path": path, "bytes": len(content.encode("utf-8")), "written": True},
        sort_keys=True,
    )


def _handle_fs_list(world: World, args: dict[str, Any]) -> str:
    entries = world.list_dir(args["path"])
    return json.dumps({"entries": entries, "count": len(entries)}, sort_keys=True)


def _handle_http_get(world: World, args: dict[str, Any]) -> str:
    return json.dumps(world.http_get(args["url"]), sort_keys=True)


def _handle_http_post(world: World, args: dict[str, Any]) -> str:
    url = args["url"]
    body = args["body"]
    route = world.http_routes.get(url)
    if route is None or "post_body" not in route:
        raise ToolFailure("not_found", f"no route registered for POST {url}")
    if not isinstance(route["post_body"], dict):
        raise ToolFailure("bad_route", f"POST {url} is misconfigured in the task definition")
    response = {"status": 201, "endpoint": url, "body": route["post_body"], "received": body}
    return json.dumps(response, sort_keys=True)


def _handle_db_query(world: World, args: dict[str, Any]) -> str:
    rows = world.query_table(args["table"], args.get("where"))
    return json.dumps({"rows": rows, "count": len(rows)}, sort_keys=True)


def _handle_calc(world: World, args: dict[str, Any]) -> str:
    return json.dumps({"expression": args["expression"], "result": evaluate(args["expression"])})


def _handle_search_docs(world: World, args: dict[str, Any]) -> str:
    hits = world.search_docs(args["query"], int(args.get("top_k", 3)))
    return json.dumps({"hits": hits, "count": len(hits)}, sort_keys=True)


def _handle_send_email(world: World, args: dict[str, Any]) -> str:
    return json.dumps(world.send_email(args["to"], args["subject"], args["body"]), sort_keys=True)


def _handle_create_event(world: World, args: dict[str, Any]) -> str:
    return json.dumps(world.create_event(args["title"], args["start"]), sort_keys=True)


def _handle_ask_user(world: World, args: dict[str, Any]) -> str:
    return json.dumps(world.ask_user(args["question"]), sort_keys=True)


HANDLERS: dict[str, Callable[[World, dict[str, Any]], str]] = {
    "fs_read": _handle_fs_read,
    "fs_write": _handle_fs_write,
    "fs_list": _handle_fs_list,
    "http_get": _handle_http_get,
    "http_post": _handle_http_post,
    "db_query": _handle_db_query,
    "calc": _handle_calc,
    "search_docs": _handle_search_docs,
    "send_email": _handle_send_email,
    "create_event": _handle_create_event,
    "ask_user": _handle_ask_user,
}


def fault_output(kind: str) -> tuple[str | None, str]:
    if kind == "timeout":
        return None, "tool timed out after 5000 ms; no output was produced"
    if kind == "http_500":
        return None, "upstream returned HTTP 500 Internal Server Error"
    return (
        '{"rows": [{"id": 1, "name": "corrupt',
        "tool returned a truncated JSON payload (invalid JSON); ignore it and retry",
    )


def parse_raw_args(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Turn whatever the model emitted into a dict, or explain why not."""

    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            return None, f"arguments are not valid JSON: {exc}"
        if not isinstance(parsed, dict):
            return None, f"arguments must decode to an object, got {type(parsed).__name__}"
        return parsed, None
    return None, f"arguments must be an object, got {type(raw).__name__}"


def call_tool(world: World, tool: str, raw_args: Any) -> ToolResult:
    """Validate, inject faults, execute and record exactly one tool call."""

    if tool not in HANDLERS:
        if tool == "finish":
            raise ToolInputError("finish is handled by the runner, not the tool layer")
        result = ToolResult(
            tool=tool,
            args={},
            ok=False,
            error=f"unknown tool {tool!r}; available: " + ", ".join(MOCK_TOOLS),
            error_kind="unknown_tool",
        )
        return world.record(result)

    args, parse_error = parse_raw_args(raw_args)
    if parse_error is not None:
        result = ToolResult(tool=tool, args={}, ok=False, error=parse_error,
                            error_kind="schema_violation")
        return world.record(result)

    violations = validate_args(TOOL_SCHEMAS[tool], args)
    if violations:
        result = ToolResult(
            tool=tool,
            args=args,
            ok=False,
            error="invalid arguments: " + "; ".join(violations),
            error_kind="schema_violation",
        )
        return world.record(result)

    ticks = world.tick(tool, args)

    try:
        plan = world.next_fault(tool)
        if plan is not None:
            output, message = fault_output(plan.kind)
            result = ToolResult(
                tool=tool,
                args=args,
                ok=False,
                output=output,
                error=message,
                error_kind=plan.kind,
                fault_injected=True,
                ticks=ticks,
            )
            return world.record(result)
        output = HANDLERS[tool](world, args)
    except SandboxError as exc:
        result = ToolResult(
            tool=tool,
            args=args,
            ok=False,
            error=f"sandbox violation: {exc}",
            error_kind="sandbox_escape",
            ticks=ticks,
        )
        return world.record(result)
    except ToolFailure as exc:
        result = ToolResult(
            tool=tool, args=args, ok=False, error=exc.message, error_kind=exc.kind, ticks=ticks
        )
        return world.record(result)
    except Exception as exc:
        result = ToolResult(
            tool=tool,
            args=args,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            error_kind="internal_error",
            ticks=ticks,
        )
        return world.record(result)

    return world.record(ToolResult(tool=tool, args=args, ok=True, output=output, ticks=ticks))


def tool_catalogue() -> list[dict[str, Any]]:
    """Machine readable description of every tool, used by `agent-bench list`."""

    catalogue = []
    for name in sorted(TOOL_SCHEMAS):
        schema = TOOL_SCHEMAS[name]
        catalogue.append(
            {
                "name": name,
                "kind": "terminal" if name == "finish" else "mock",
                "returns_json": name in JSON_TOOLS,
                "required": list(schema.get("required", [])),
                "properties": sorted(schema.get("properties", {})),
            }
        )
    return catalogue
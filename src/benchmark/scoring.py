"""Deterministic scoring of tool calls against expectations.

No LLM judging. Comparison rules:

* Tool names must match exactly (case-insensitively when
  ``require_exact_tool_names`` is false).
* Arguments are compared as JSON values. Strings are trimmed of surrounding
  whitespace when ``trim_string_whitespace`` is true. ``int`` and ``float``
  are both JSON numbers, so ``5`` equals ``5.0``; but ``"5"`` (string) never
  equals ``5`` (number) unless ``allow_argument_coercion`` is true. Booleans
  are never numbers.
* ``require_exact_arguments`` true means every argument object — top level
  and nested — must match key-for-key; false means the expected keys must be
  a subset at every level (extra keys are ignored).
* Expected values may be **matchers** instead of literals (see
  :data:`MATCHER_OPS`), e.g. ``{"$ci": "bench press"}`` or
  ``{"$range": [100, 200]}``.
* ``require_order`` true means expected calls must appear in order; false
  means any order.
* ``allow_extra_tool_calls`` false means any unexpected call fails the run,
  except calls to tools listed in ``allowed_extra_tools``.
* ``validate_arguments_schema`` true also validates every emitted call's
  arguments against the offered tool's JSON Schema (what a server-side
  validator such as Zod would reject).

Matching is optimal, not greedy: when several expected calls could pair with
several actual calls, the scorer finds the assignment that satisfies the most
expectations, so a correct answer never fails because of pairing order.

Any single violation fails the whole run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .schema import (
    ArgumentError,
    ExpectedToolCall,
    FailureType,
    SchemaError,
    ScoringConfig,
    ScoringResult,
    ToolCall,
    WrongToolCall,
)

# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------

#: Supported matcher operators. A matcher is a JSON object whose keys all
#: start with ``$`` and whose first key is one of these:
#:
#: ``{"$any": true}``             key present, any value
#: ``{"$absent": true}``          key must NOT be present
#: ``{"$optional": <value>}``     key may be absent; if present must match
#: ``{"$ci": "text"}``            string equal ignoring case and runs of spaces
#: ``{"$contains": "t" | [..]}``  string contains every fragment (ignoring case)
#: ``{"$regex": "pattern"}``      string fully matches the regex
#: ``{"$one_of": [v1, v2]}``      matches any alternative (alternatives may be
#:                                 matchers)
#: ``{"$range": [lo, hi]}``       number with lo <= x <= hi (null = unbounded)
#: ``{"$unordered": [v1, v2]}``   array with exactly these items in any order
#: ``{"$includes": [v1, v2]}``    array containing (at least) these items
MATCHER_OPS = ("$any", "$absent", "$optional", "$ci", "$contains", "$regex",
               "$one_of", "$range", "$unordered", "$includes")


def is_matcher(value: Any) -> bool:
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(k, str) and k.startswith("$") for k in value))


def validate_matchers(value: Any, path: str = "") -> list[str]:
    """Return human-readable problems with matcher syntax inside ``value``."""
    problems: list[str] = []
    if is_matcher(value):
        op = next(iter(value))
        where = path or "<root>"
        if op not in MATCHER_OPS:
            return [f"{where}: unknown matcher {op!r} "
                    f"(supported: {', '.join(MATCHER_OPS)})"]
        arg = value[op]
        if op in ("$any", "$absent") and arg is not True:
            problems.append(f"{where}: {op} must be true")
        elif op == "$ci" and not isinstance(arg, str):
            problems.append(f"{where}: $ci needs a string")
        elif op == "$contains" and not (
                isinstance(arg, str) or (isinstance(arg, list) and arg
                                         and all(isinstance(a, str) for a in arg))):
            problems.append(f"{where}: $contains needs a string or list of strings")
        elif op == "$regex":
            if not isinstance(arg, str):
                problems.append(f"{where}: $regex needs a string")
            else:
                try:
                    re.compile(arg)
                except re.error as exc:
                    problems.append(f"{where}: bad $regex: {exc}")
        elif op in ("$one_of", "$unordered", "$includes"):
            if not isinstance(arg, list) or (op == "$one_of" and not arg):
                problems.append(f"{where}: {op} needs a "
                                f"{'non-empty ' if op == '$one_of' else ''}list")
            else:
                for i, alt in enumerate(arg):
                    problems.extend(validate_matchers(alt, f"{where}|{i}"))
        elif op == "$range":
            if (not isinstance(arg, list) or len(arg) != 2 or not all(
                    a is None or (isinstance(a, (int, float))
                                  and not isinstance(a, bool)) for a in arg)):
                problems.append(f"{where}: $range needs [lo, hi] numbers or nulls")
        elif op == "$optional":
            problems.extend(validate_matchers(arg, where))
        return problems
    if isinstance(value, dict):
        for k, v in value.items():
            problems.extend(validate_matchers(v, f"{path}.{k}" if path else k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            problems.extend(validate_matchers(v, f"{path}[{i}]"))
    return problems


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------


def _normalize(value: Any, cfg: ScoringConfig) -> Any:
    if isinstance(value, str) and cfg.trim_string_whitespace:
        return value.strip()
    if isinstance(value, dict):
        return {k: _normalize(v, cfg) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v, cfg) for v in value]
    return value


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _coerce(expected: Any, actual: Any) -> Any:
    """Best-effort coercion of ``actual`` toward the JSON type of
    ``expected``. Only used when ``allow_argument_coercion`` is true."""
    if isinstance(expected, bool) and isinstance(actual, str):
        lowered = actual.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        return actual
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) \
            and isinstance(actual, str):
        try:
            return float(actual) if "." in actual or "e" in actual.lower() \
                else int(actual)
        except ValueError:
            return actual
    if isinstance(expected, str) and isinstance(actual, (int, float, bool)):
        if isinstance(actual, bool):
            return "true" if actual else "false"
        return str(actual)
    return actual


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


class _Matcher:
    """Recursive expected-vs-actual comparison that records granular errors."""

    def __init__(self, tool: str, cfg: ScoringConfig):
        self.tool = tool
        self.cfg = cfg
        self.subset = not cfg.require_exact_arguments

    def _err(self, errors: list[ArgumentError], path: str, kind: str,
             expected: Any = None, actual: Any = None) -> None:
        errors.append(ArgumentError(tool_name=self.tool, argument_path=path,
                                    error_type=kind, expected=expected,
                                    actual=actual))

    def value(self, expected: Any, actual: Any, path: str,
              errors: list[ArgumentError]) -> bool:
        if is_matcher(expected):
            return self._matcher(expected, actual, path, errors)
        if self.cfg.allow_argument_coercion:
            actual = _coerce(expected, actual)
        e_type, a_type = _json_type(expected), _json_type(actual)
        if e_type != a_type:
            self._err(errors, path, "type_mismatch", expected, actual)
            return False
        if e_type == "object":
            return self.obj(expected, actual, path, errors)
        if e_type == "array":
            if len(expected) != len(actual):
                self._err(errors, path, "wrong_value", expected, actual)
                return False
            ok = True
            for i, (e, a) in enumerate(zip(expected, actual)):
                ok &= self.value(e, a, f"{path}[{i}]", errors)
            return ok
        if e_type == "number":
            same = float(expected) == float(actual)
        else:
            same = expected == actual
        if not same:
            self._err(errors, path, "wrong_value", expected, actual)
        return same

    def obj(self, expected: dict[str, Any], actual: dict[str, Any],
            path: str, errors: list[ArgumentError]) -> bool:
        ok = True
        for key, e_val in expected.items():
            child = f"{path}.{key}" if path else key
            op = next(iter(e_val)) if is_matcher(e_val) else None
            if op == "$absent":
                if key in actual:
                    self._err(errors, child, "unexpected_argument",
                              actual=actual[key])
                    ok = False
                continue
            if key not in actual:
                if op == "$optional":
                    continue
                self._err(errors, child, "missing_argument", expected=e_val)
                ok = False
                continue
            ok &= self.value(e_val, actual[key], child, errors)
        if not self.subset:
            for key in actual:
                if key not in expected:
                    self._err(errors, f"{path}.{key}" if path else key,
                              "extra_argument", actual=actual[key])
                    ok = False
        return ok

    def _matcher(self, spec: dict[str, Any], actual: Any, path: str,
                 errors: list[ArgumentError]) -> bool:
        op = next(iter(spec))
        arg = spec[op]
        if op in ("$any", "$absent"):
            # Presence is decided in obj(); reaching here means present.
            if op == "$absent":
                self._err(errors, path, "unexpected_argument", actual=actual)
                return False
            return True
        if op == "$optional":
            return self.value(arg, actual, path, errors)
        if op == "$one_of":
            for alt in arg:
                if self.value(alt, actual, path, []):
                    return True
            self._err(errors, path, "wrong_value", spec, actual)
            return False
        if op in ("$unordered", "$includes"):
            if not isinstance(actual, list):
                self._err(errors, path, "type_mismatch", spec, actual)
                return False
            if op == "$unordered" and len(actual) != len(arg):
                self._err(errors, path, "wrong_value", spec, actual)
                return False
            pairs = _bipartite(len(arg), len(actual),
                               lambda i, j: self.value(arg[i], actual[j], path, []))
            if len(pairs) == len(arg):
                return True
            self._err(errors, path, "wrong_value", spec, actual)
            return False
        if op == "$range":
            lo, hi = arg
            ok = (_json_type(actual) == "number"
                  and (lo is None or actual >= lo)
                  and (hi is None or actual <= hi))
            if not ok:
                kind = ("type_mismatch" if _json_type(actual) != "number"
                        else "wrong_value")
                self._err(errors, path, kind, spec, actual)
            return ok
        # String matchers.
        if not isinstance(actual, str):
            self._err(errors, path, "type_mismatch", spec, actual)
            return False
        if op == "$ci":
            ok = _squash(actual) == _squash(arg)
        elif op == "$contains":
            fragments = [arg] if isinstance(arg, str) else arg
            ok = all(_squash(f) in _squash(actual) for f in fragments)
        else:  # $regex
            ok = re.fullmatch(arg, actual) is not None
        if not ok:
            self._err(errors, path, "wrong_value", spec, actual)
        return ok


def compare_arguments(tool: str, expected: dict[str, Any],
                      actual: dict[str, Any],
                      cfg: ScoringConfig) -> list[ArgumentError]:
    """Compare one call's arguments; empty list means they match."""
    errors: list[ArgumentError] = []
    _Matcher(tool, cfg).obj(_normalize(expected, cfg),
                            _normalize(actual, cfg), "", errors)
    return errors


def _arguments_match(expected: ToolCall, actual: ToolCall,
                     cfg: ScoringConfig) -> bool:
    return not compare_arguments(expected.name, expected.arguments,
                                 actual.arguments, cfg)


def _names_match(expected: str, actual: str, cfg: ScoringConfig) -> bool:
    if cfg.require_exact_tool_names:
        return expected == actual
    return expected.lower() == actual.lower()


def _variants(expected: ToolCall) -> list[ToolCall]:
    if isinstance(expected, ExpectedToolCall):
        return expected.variants()
    return [expected]


def _call_matches(expected: ToolCall, actual: ToolCall,
                  cfg: ScoringConfig) -> bool:
    """The actual call satisfies the expectation or one of its alternatives."""
    return any(_names_match(v.name, actual.name, cfg)
               and _arguments_match(v, actual, cfg)
               for v in _variants(expected))


def _name_variant(expected: ToolCall, actual: ToolCall,
                  cfg: ScoringConfig) -> ToolCall | None:
    """The expectation variant naming the same tool as ``actual``, if any."""
    return next((v for v in _variants(expected)
                 if _names_match(v.name, actual.name, cfg)), None)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _schema_validator_factory() -> Callable[[dict[str, Any]], Any]:
    from jsonschema import Draft7Validator, FormatChecker

    checker = FormatChecker()

    @checker.checks("uuid", raises=ValueError)
    def _is_uuid(value: Any) -> bool:  # Draft 7's checker lacks uuid
        if not isinstance(value, str):
            return True
        return re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value) is not None

    cache: dict[str, Any] = {}

    def make(schema: dict[str, Any]) -> Any:
        key = json.dumps(schema, sort_keys=True)
        if key not in cache:
            cache[key] = Draft7Validator(schema, format_checker=checker)
        return cache[key]

    return make


_make_validator: Callable[[dict[str, Any]], Any] | None = None


def schema_errors_for(call: ToolCall,
                      schema: dict[str, Any]) -> list[SchemaError]:
    global _make_validator
    if _make_validator is None:
        _make_validator = _schema_validator_factory()
    validator = _make_validator(schema)
    out: list[SchemaError] = []
    for err in sorted(validator.iter_errors(call.arguments),
                      key=lambda e: list(e.absolute_path)):
        path = "".join(f"[{p}]" if isinstance(p, int) else f".{p}"
                       for p in err.absolute_path).lstrip(".")
        out.append(SchemaError(tool_name=call.name, argument_path=path,
                               message=err.message[:300]))
    return out


# ---------------------------------------------------------------------------
# Run-level scoring
# ---------------------------------------------------------------------------


def score_tool_calls(expected: list[ToolCall], actual: list[ToolCall],
                     cfg: ScoringConfig, *, error: str | None = None,
                     timed_out: bool = False,
                     protocol_error: str | None = None,
                     tool_schemas: dict[str, dict[str, Any]] | None = None,
                     ) -> ScoringResult:
    """Score one run. Deterministic; every violation fails the run.

    ``tool_schemas`` maps offered tool names to their JSON Schema; it is
    consulted only when ``cfg.validate_arguments_schema`` is true.
    """
    if timed_out:
        return ScoringResult(success=False, reason="Timed out.")
    if error is not None:
        return ScoringResult(success=False, reason=f"Provider error: {error}")
    if protocol_error is not None:
        return ScoringResult(success=False, protocol_error=protocol_error,
                             reason=f"Protocol error: {protocol_error}")

    schema_errs: list[SchemaError] = []
    if cfg.validate_arguments_schema and tool_schemas:
        for call in actual:
            schema = tool_schemas.get(call.name)
            if schema is not None:
                schema_errs.extend(schema_errors_for(call, schema))

    if not expected:
        unexpected = [c.name for c in actual
                      if c.name not in cfg.allowed_extra_tools]
        if unexpected:
            return ScoringResult(
                success=False, extra_tool_calls=unexpected,
                schema_errors=schema_errs,
                reason="No tool call expected, but the model called: "
                       f"{', '.join(unexpected)}.",
            )
        return _finalize([], [], [], [], [], schema_errs)

    if not actual:
        return ScoringResult(
            success=False,
            missing_tool_calls=[c.name for c in expected],
            reason="Model produced no tool calls (text only).",
        )

    if cfg.require_order:
        return _score_ordered(expected, actual, cfg, schema_errs)
    return _score_unordered(expected, actual, cfg, schema_errs)


def _finalize(missing: list[str], wrong: list[WrongToolCall],
              arg_errors: list[ArgumentError], extras: list[str],
              ordering: list[str],
              schema_errs: list[SchemaError]) -> ScoringResult:
    success = not (missing or wrong or arg_errors or extras or ordering
                   or schema_errs)
    if success:
        reason = "Passed."
    else:
        parts: list[str] = []
        if missing:
            parts.append(f"missing tool calls: {', '.join(missing)}")
        if wrong:
            parts.append(
                "wrong tool names: " + ", ".join(
                    f"expected {w.expected} got {w.actual}" for w in wrong)
            )
        if schema_errs:
            parts.append(
                "schema violations: " + "; ".join(
                    f"{e.tool_name}.{e.argument_path or '<root>'} ({e.message})"
                    for e in schema_errs[:5])
                + (f"; +{len(schema_errs) - 5} more" if len(schema_errs) > 5 else "")
            )
        if arg_errors:
            parts.append(
                "argument errors: " + "; ".join(
                    f"{e.tool_name}.{e.argument_path} ({e.error_type})"
                    for e in arg_errors)
            )
        if extras:
            parts.append(f"extra tool calls: {', '.join(extras)}")
        if ordering:
            parts.append("ordering violations: " + "; ".join(ordering))
        reason = "Failed — " + "; ".join(parts) + "."
    return ScoringResult(
        success=success, missing_tool_calls=missing, wrong_tool_calls=wrong,
        argument_errors=arg_errors, extra_tool_calls=extras,
        ordering_errors=ordering, schema_errors=schema_errs, reason=reason,
    )


def _bipartite(n_left: int, n_right: int,
               ok: Callable[[int, int], bool]) -> dict[int, int]:
    """Maximum bipartite matching left→right (Kuhn's algorithm).

    Call lists and argument arrays are tiny, so the O(V·E) augmenting-path
    search is instant and — unlike greedy first-fit — never strands a match
    another pairing could have made.
    """
    adj = [[j for j in range(n_right) if ok(i, j)] for i in range(n_left)]
    owner: dict[int, int] = {}  # actual index -> expected index

    def augment(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(n_left):
        augment(i, set())
    return {i: j for j, i in owner.items()}


def _leftovers(expected: list[ToolCall], actual: list[ToolCall],
               unmatched: list[int], used: set[int], cfg: ScoringConfig,
               ) -> tuple[list[str], list[WrongToolCall], list[ArgumentError],
                          list[str]]:
    """Diagnose unmatched expected calls and unused actual calls."""
    arg_errors: list[ArgumentError] = []
    still_missing: list[int] = []
    for i in unmatched:
        exp = expected[i]
        hit = next((j for j, act in enumerate(actual)
                    if j not in used and _name_variant(exp, act, cfg) is not None),
                   None)
        if hit is None:
            still_missing.append(i)
            continue
        used.add(hit)
        variant = _name_variant(exp, actual[hit], cfg)
        arg_errors.extend(compare_arguments(variant.name, variant.arguments,
                                            actual[hit].arguments, cfg))

    allowed = set(cfg.allowed_extra_tools)
    leftover = [j for j in range(len(actual))
                if j not in used and actual[j].name not in allowed]
    wrong: list[WrongToolCall] = []
    missing: list[str] = []
    extras: list[str] = []
    if cfg.allow_extra_tool_calls:
        # Extras are permitted, so an unmatched actual call is not evidence
        # of a wrong tool choice — unmatched expected calls are just missing.
        missing.extend(expected[i].name for i in still_missing)
    else:
        pair_count = min(len(still_missing), len(leftover))
        for k in range(pair_count):
            wrong.append(WrongToolCall(expected=expected[still_missing[k]].name,
                                       actual=actual[leftover[k]].name))
        missing.extend(expected[i].name for i in still_missing[pair_count:])
        extras = [actual[j].name for j in leftover[pair_count:]]
    return missing, wrong, arg_errors, extras


def _score_unordered(expected: list[ToolCall], actual: list[ToolCall],
                     cfg: ScoringConfig,
                     schema_errs: list[SchemaError]) -> ScoringResult:
    pairs = _bipartite(len(expected), len(actual),
                       lambda i, j: _call_matches(expected[i], actual[j], cfg))
    used = set(pairs.values())
    unmatched = [i for i in range(len(expected)) if i not in pairs]
    missing, wrong, arg_errors, extras = _leftovers(
        expected, actual, unmatched, used, cfg)
    return _finalize(missing, wrong, arg_errors, extras, [], schema_errs)


def _score_ordered(expected: list[ToolCall], actual: list[ToolCall],
                   cfg: ScoringConfig,
                   schema_errs: list[SchemaError]) -> ScoringResult:
    # Expected calls must appear as an ordered subsequence of the actual
    # calls. Earliest-match greedy is optimal for subsequence search.
    used: set[int] = set()
    unmatched: list[int] = []
    pos = 0
    for i, exp in enumerate(expected):
        hit = next((j for j in range(pos, len(actual))
                    if _call_matches(exp, actual[j], cfg)), None)
        if hit is None:
            unmatched.append(i)
        else:
            used.add(hit)
            pos = hit + 1

    # A call that is correct but out of place is an ordering violation, not
    # a missing or extra call.
    ordering: list[str] = []
    still: list[int] = []
    for i in unmatched:
        exp = expected[i]
        anywhere = next((j for j, act in enumerate(actual)
                         if j not in used and _call_matches(exp, act, cfg)), None)
        if anywhere is None:
            still.append(i)
        else:
            used.add(anywhere)
            ordering.append(
                f"call to {exp.name} occurred out of the required order")
    if ordering and not still:
        exp_names = [c.name for c in expected]
        act_names = [c.name for c in actual]
        ordering = [f"expected order [{', '.join(exp_names)}] but got "
                    f"[{', '.join(act_names)}]"]

    missing, wrong, arg_errors, extras = _leftovers(
        expected, actual, still, used, cfg)
    return _finalize(missing, wrong, arg_errors, extras, ordering, schema_errs)


# ---------------------------------------------------------------------------
# Failure classification (one primary type per failed run)
# ---------------------------------------------------------------------------


def classify_failure(result: ScoringResult, *, error: str | None = None,
                     timed_out: bool = False) -> FailureType | None:
    """Return the primary failure type for a failed run, or None on success.

    Precedence: timeout > provider error > protocol error > missing call >
    wrong name > schema violation > wrong argument > extra call > ordering.
    Each failed run counts exactly once in summary statistics.
    """
    if result.success:
        return None
    if timed_out:
        return FailureType.TIMEOUT
    if error is not None:
        return FailureType.PROVIDER_ERROR
    if result.protocol_error:
        return FailureType.PROTOCOL_ERROR
    if result.missing_tool_calls:
        return FailureType.MISSING_TOOL_CALL
    if result.wrong_tool_calls:
        return FailureType.WRONG_TOOL_NAME
    if result.schema_errors:
        return FailureType.SCHEMA_VIOLATION
    if result.argument_errors:
        return FailureType.WRONG_ARGUMENT
    if result.extra_tool_calls:
        return FailureType.EXTRA_TOOL_CALL
    if result.ordering_errors:
        return FailureType.ORDERING_ERROR
    # Fallback — should not happen, but never leave a failure unclassified.
    return FailureType.WRONG_ARGUMENT

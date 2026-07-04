"""Deterministic scoring of tool calls against expectations.

No LLM judging. Comparison rules:

* Tool names must match exactly (case-insensitively when
  ``require_exact_tool_names`` is false).
* Arguments are compared as JSON values. Strings are trimmed of surrounding
  whitespace when ``trim_string_whitespace`` is true. ``int`` and ``float``
  are both JSON numbers, so ``5`` equals ``5.0``; but ``"5"`` (string) never
  equals ``5`` (number) unless ``allow_argument_coercion`` is true. Booleans
  are never numbers.
* ``require_exact_arguments`` true means the argument object must match
  key-for-key; false means the expected arguments must be a subset.
* ``require_order`` true means expected calls must appear in order; false
  means any order.
* ``allow_extra_tool_calls`` false means any unexpected call fails the run.

Any single violation fails the whole run.
"""

from __future__ import annotations

from typing import Any

from .schema import (
    ArgumentError,
    FailureType,
    ScoringConfig,
    ScoringResult,
    ToolCall,
    WrongToolCall,
)

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
    if isinstance(expected, str) and isinstance(actual, (int, float, bool)) :
        if isinstance(actual, bool):
            return "true" if actual else "false"
        return str(actual)
    return actual


def _values_equal(expected: Any, actual: Any, cfg: ScoringConfig) -> bool:
    if cfg.allow_argument_coercion:
        actual = _coerce(expected, actual)
    e_type, a_type = _json_type(expected), _json_type(actual)
    if e_type != a_type:
        return False
    if e_type == "number":
        return float(expected) == float(actual)
    if e_type == "object":
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(_values_equal(expected[k], actual[k], cfg) for k in expected)
    if e_type == "array":
        if len(expected) != len(actual):
            return False
        return all(_values_equal(e, a, cfg) for e, a in zip(expected, actual))
    return expected == actual


def _diff_value(tool: str, path: str, expected: Any, actual: Any,
                cfg: ScoringConfig, errors: list[ArgumentError]) -> None:
    """Record granular argument errors for a mismatched value."""
    checked = _coerce(expected, actual) if cfg.allow_argument_coercion else actual
    e_type, a_type = _json_type(expected), _json_type(checked)
    if e_type != a_type:
        errors.append(ArgumentError(
            tool_name=tool, argument_path=path, error_type="type_mismatch",
            expected=expected, actual=actual,
        ))
        return
    if e_type == "object":
        for key in expected:
            child = f"{path}.{key}" if path else key
            if key not in checked:
                errors.append(ArgumentError(
                    tool_name=tool, argument_path=child,
                    error_type="missing_argument", expected=expected[key],
                ))
            elif not _values_equal(expected[key], checked[key], cfg):
                _diff_value(tool, child, expected[key], checked[key], cfg, errors)
        for key in checked:
            if key not in expected:
                errors.append(ArgumentError(
                    tool_name=tool, argument_path=f"{path}.{key}" if path else key,
                    error_type="extra_argument", actual=checked[key],
                ))
        return
    if e_type == "array":
        if len(expected) != len(checked):
            errors.append(ArgumentError(
                tool_name=tool, argument_path=path, error_type="wrong_value",
                expected=expected, actual=actual,
            ))
            return
        for i, (e, a) in enumerate(zip(expected, checked)):
            if not _values_equal(e, a, cfg):
                _diff_value(tool, f"{path}[{i}]", e, a, cfg, errors)
        return
    errors.append(ArgumentError(
        tool_name=tool, argument_path=path, error_type="wrong_value",
        expected=expected, actual=actual,
    ))


def compare_arguments(tool: str, expected: dict[str, Any],
                      actual: dict[str, Any],
                      cfg: ScoringConfig) -> list[ArgumentError]:
    """Compare one call's arguments; empty list means they match."""
    expected = _normalize(expected, cfg)
    actual = _normalize(actual, cfg)
    errors: list[ArgumentError] = []
    for key, e_val in expected.items():
        if key not in actual:
            errors.append(ArgumentError(
                tool_name=tool, argument_path=key,
                error_type="missing_argument", expected=e_val,
            ))
        elif not _values_equal(e_val, actual[key], cfg):
            _diff_value(tool, key, e_val, actual[key], cfg, errors)
    if cfg.require_exact_arguments:
        for key in actual:
            if key not in expected:
                errors.append(ArgumentError(
                    tool_name=tool, argument_path=key,
                    error_type="extra_argument", actual=actual[key],
                ))
    return errors


def _arguments_match(expected: ToolCall, actual: ToolCall,
                     cfg: ScoringConfig) -> bool:
    return not compare_arguments(expected.name, expected.arguments,
                                 actual.arguments, cfg)


def _names_match(expected: str, actual: str, cfg: ScoringConfig) -> bool:
    if cfg.require_exact_tool_names:
        return expected == actual
    return expected.lower() == actual.lower()


# ---------------------------------------------------------------------------
# Run-level scoring
# ---------------------------------------------------------------------------


def score_tool_calls(expected: list[ToolCall], actual: list[ToolCall],
                     cfg: ScoringConfig, *, error: str | None = None,
                     timed_out: bool = False) -> ScoringResult:
    """Score one run. Deterministic; every violation fails the run."""
    if timed_out:
        return ScoringResult(success=False, reason="Timed out.")
    if error is not None:
        return ScoringResult(success=False, reason=f"Provider error: {error}")

    if not expected:
        if actual:
            names = [c.name for c in actual]
            return ScoringResult(
                success=False, extra_tool_calls=names,
                reason=f"No tool call expected, but the model called: "
                       f"{', '.join(names)}.",
            )
        return ScoringResult(success=True, reason="Passed.")

    if not actual:
        return ScoringResult(
            success=False,
            missing_tool_calls=[c.name for c in expected],
            reason="Model produced no tool calls (text only).",
        )

    if cfg.require_order:
        result = _score_ordered(expected, actual, cfg)
    else:
        result = _score_unordered(expected, actual, cfg)
    return result


def _finalize(missing: list[str], wrong: list[WrongToolCall],
              arg_errors: list[ArgumentError], extras: list[str],
              ordering: list[str]) -> ScoringResult:
    success = not (missing or wrong or arg_errors or extras or ordering)
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
        ordering_errors=ordering, reason=reason,
    )


def _score_unordered(expected: list[ToolCall], actual: list[ToolCall],
                     cfg: ScoringConfig) -> ScoringResult:
    used = [False] * len(actual)
    unmatched_expected: list[ToolCall] = []
    arg_errors: list[ArgumentError] = []

    # Pass 1: exact (name + arguments) matches.
    for exp in expected:
        hit = next(
            (i for i, act in enumerate(actual)
             if not used[i] and _names_match(exp.name, act.name, cfg)
             and _arguments_match(exp, act, cfg)),
            None,
        )
        if hit is None:
            unmatched_expected.append(exp)
        else:
            used[hit] = True

    # Pass 2: same tool name, wrong arguments → argument errors.
    still_missing: list[ToolCall] = []
    for exp in unmatched_expected:
        hit = next(
            (i for i, act in enumerate(actual)
             if not used[i] and _names_match(exp.name, act.name, cfg)),
            None,
        )
        if hit is None:
            still_missing.append(exp)
        else:
            used[hit] = True
            arg_errors.extend(
                compare_arguments(exp.name, exp.arguments,
                                  actual[hit].arguments, cfg)
            )

    # Pass 3: pair leftover expected with leftover wrong-named actual calls.
    leftover_actual = [actual[i] for i in range(len(actual)) if not used[i]]
    wrong: list[WrongToolCall] = []
    missing: list[str] = []
    if cfg.allow_extra_tool_calls:
        # Extras are permitted, so an unmatched actual call is not evidence
        # of a wrong tool choice — unmatched expected calls are just missing.
        missing.extend(c.name for c in still_missing)
        extras: list[str] = []
    else:
        pair_count = min(len(still_missing), len(leftover_actual))
        for i in range(pair_count):
            wrong.append(WrongToolCall(expected=still_missing[i].name,
                                       actual=leftover_actual[i].name))
        missing.extend(c.name for c in still_missing[pair_count:])
        extras = [c.name for c in leftover_actual[pair_count:]]

    return _finalize(missing, wrong, arg_errors, extras, [])


def _score_ordered(expected: list[ToolCall], actual: list[ToolCall],
                   cfg: ScoringConfig) -> ScoringResult:
    exp_names = [c.name for c in expected]
    act_names = [c.name for c in actual]

    if not cfg.allow_extra_tool_calls:
        # The name sequences must be identical, then arguments positional.
        if exp_names == act_names:
            arg_errors: list[ArgumentError] = []
            for exp, act in zip(expected, actual):
                arg_errors.extend(
                    compare_arguments(exp.name, exp.arguments, act.arguments, cfg)
                )
            return _finalize([], [], arg_errors, [], [])
        if sorted(exp_names) == sorted(act_names):
            ordering = [
                f"expected order [{', '.join(exp_names)}] but got "
                f"[{', '.join(act_names)}]"
            ]
            return _finalize([], [], [], [], ordering)
        # Different call sets entirely — reuse unordered diagnostics.
        return _score_unordered(expected, actual, cfg)

    # Extras allowed: expected must appear as a subsequence of actual.
    used = [False] * len(actual)
    pos = 0
    unmatched: list[ToolCall] = []
    for exp in expected:
        hit = next(
            (i for i in range(pos, len(actual))
             if not used[i] and _names_match(exp.name, actual[i].name, cfg)
             and _arguments_match(exp, actual[i], cfg)),
            None,
        )
        if hit is None:
            unmatched.append(exp)
        else:
            used[hit] = True
            pos = hit + 1

    if not unmatched:
        return _finalize([], [], [], [], [])

    # Distinguish "present but out of order / bad args" from "missing".
    missing: list[str] = []
    arg_errors = []
    ordering = []
    for exp in unmatched:
        anywhere = next(
            (i for i, act in enumerate(actual)
             if _names_match(exp.name, act.name, cfg)
             and _arguments_match(exp, act, cfg)),
            None,
        )
        if anywhere is not None:
            ordering.append(
                f"call to {exp.name} occurred out of the required order"
            )
            continue
        name_hit = next(
            (i for i, act in enumerate(actual)
             if not used[i] and _names_match(exp.name, act.name, cfg)),
            None,
        )
        if name_hit is not None:
            used[name_hit] = True
            arg_errors.extend(
                compare_arguments(exp.name, exp.arguments,
                                  actual[name_hit].arguments, cfg)
            )
        else:
            missing.append(exp.name)
    return _finalize(missing, [], arg_errors, [], ordering)


# ---------------------------------------------------------------------------
# Failure classification (one primary type per failed run)
# ---------------------------------------------------------------------------


def classify_failure(result: ScoringResult, *, error: str | None = None,
                     timed_out: bool = False) -> FailureType | None:
    """Return the primary failure type for a failed run, or None on success.

    Precedence: timeout > provider error > missing call > wrong name >
    wrong argument > extra call > ordering. Each failed run counts exactly
    once in summary statistics.
    """
    if result.success:
        return None
    if timed_out:
        return FailureType.TIMEOUT
    if error is not None:
        return FailureType.PROVIDER_ERROR
    if result.missing_tool_calls:
        return FailureType.MISSING_TOOL_CALL
    if result.wrong_tool_calls:
        return FailureType.WRONG_TOOL_NAME
    if result.argument_errors:
        return FailureType.WRONG_ARGUMENT
    if result.extra_tool_calls:
        return FailureType.EXTRA_TOOL_CALL
    if result.ordering_errors:
        return FailureType.ORDERING_ERROR
    # Fallback — should not happen, but never leave a failure unclassified.
    return FailureType.WRONG_ARGUMENT

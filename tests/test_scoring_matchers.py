"""Optimal matching, recursive subset mode, matchers, allowed extras, and
JSON-schema validation."""

import pytest

from benchmark.schema import Sample, ScoringConfig, ToolCall
from benchmark.scoring import (
    classify_failure, score_tool_calls, validate_matchers,
)


def call(name, /, **arguments):
    return ToolCall(name=name, arguments=arguments)


def cfg(**overrides):
    return ScoringConfig(**overrides)


# ---------------------------------------------------------------- regressions


def test_unordered_subset_matching_is_optimal_not_greedy():
    # Greedy first-fit pairs the richer actual call with the looser
    # expectation and then strands the second one.
    expected = [call("f", x=1), call("f", x=1, y=2)]
    actual = [call("f", x=1, y=2), call("f", x=1)]
    result = score_tool_calls(expected, actual,
                              cfg(require_exact_arguments=False,
                                  require_order=False))
    assert result.success, result.reason


def test_case_insensitive_names_still_enforce_order():
    expected = [call("first", a=1), call("second", b=2)]
    actual = [call("SECOND", b=2), call("First", a=1)]
    result = score_tool_calls(expected, actual,
                              cfg(require_exact_tool_names=False))
    assert not result.success
    assert result.ordering_errors
    assert classify_failure(result).value == "ordering_error"


def test_case_insensitive_names_in_order_pass():
    expected = [call("first", a=1), call("second", b=2)]
    actual = [call("First", a=1), call("SECOND", b=2)]
    assert score_tool_calls(expected, actual,
                            cfg(require_exact_tool_names=False)).success


def test_subset_mode_applies_to_nested_objects():
    expected = [call("log", sets=[{"weight": 185, "reps": 5}])]
    actual = [call("log", sets=[{"weight": 185, "reps": 5, "setType": "working"}],
                   title="Push")]
    assert score_tool_calls(expected, actual,
                            cfg(require_exact_arguments=False)).success
    # Exact mode still rejects the nested extra key.
    exact = score_tool_calls(expected, [call("log", sets=[{"weight": 185, "reps": 5,
                                                           "setType": "working"}])],
                             cfg())
    assert not exact.success
    assert exact.argument_errors[0].argument_path == "sets[0].setType"


def test_ordered_wrong_value_reports_positional_argument_error():
    expected = [call("a", v=1), call("b", v=2)]
    actual = [call("a", v=1), call("b", v=3)]
    result = score_tool_calls(expected, actual, cfg())
    assert not result.success
    assert [(e.tool_name, e.argument_path, e.error_type)
            for e in result.argument_errors] == [("b", "v", "wrong_value")]


# ------------------------------------------------------------------ matchers


@pytest.mark.parametrize("spec, value, ok", [
    ({"$any": True}, "anything", True),
    ({"$ci": "Bench Press"}, "bench   press", True),
    ({"$ci": "Bench Press"}, "incline bench press", False),
    ({"$contains": ["knee", "squat"]}, "Left knee hurts on deep Squats", True),
    ({"$contains": "knee"}, "shoulder", False),
    ({"$regex": "(?i)(barbell )?bench( press)?"}, "Barbell Bench Press", True),
    ({"$regex": "(?i)(barbell )?bench( press)?"}, "Incline Bench", False),
    ({"$one_of": ["lb", {"$ci": "pounds"}]}, "Pounds", True),
    ({"$one_of": ["lb", "kg"]}, "stone", False),
    ({"$range": [100, 200]}, 140, True),
    ({"$range": [100, None]}, 99, False),
    ({"$range": [0, 10]}, "5", False),
    ({"$ci": "x"}, 5, False),
])
def test_matcher_values(spec, value, ok):
    result = score_tool_calls([call("t", v=spec)], [call("t", v=value)], cfg())
    assert result.success is ok, result.reason


def test_absent_matcher_rejects_present_key():
    expected = [call("timer", action="start", seconds={"$absent": True})]
    assert score_tool_calls(expected, [call("timer", action="start")], cfg()).success
    result = score_tool_calls(expected,
                              [call("timer", action="start", seconds=90)], cfg())
    assert not result.success
    assert result.argument_errors[0].error_type == "unexpected_argument"


def test_optional_matcher_allows_absence_but_checks_presence():
    expected = [call("log", weight=185, weightUnit={"$optional": "lb"})]
    mode = cfg(require_exact_arguments=False)
    assert score_tool_calls(expected, [call("log", weight=185)], mode).success
    assert score_tool_calls(expected, [call("log", weight=185, weightUnit="lb")],
                            mode).success
    assert not score_tool_calls(expected,
                                [call("log", weight=185, weightUnit="kg")],
                                mode).success


def test_any_still_requires_presence():
    result = score_tool_calls([call("t", intro={"$any": True})], [call("t")], cfg())
    assert not result.success
    assert result.argument_errors[0].error_type == "missing_argument"


def test_validate_matchers_flags_bad_syntax():
    problems = validate_matchers({"a": {"$nope": 1}, "b": {"$regex": "("},
                                  "c": [{"$range": [1]}]})
    assert len(problems) == 3


def test_sample_rejects_unknown_matcher():
    with pytest.raises(ValueError, match="unknown matcher"):
        Sample.model_validate({
            "id": "0001_bad", "category": "x", "difficulty": "easy",
            "prompt": "p",
            "tools": [{"type": "function", "function": {
                "name": "t", "description": "",
                "parameters": {"type": "object", "properties": {}}}}],
            "expected_tool_calls": [{"name": "t", "arguments": {"v": {"$bogus": 1}}}],
        })


# ------------------------------------------------------------ allowed extras


def test_allowed_extra_tools_are_free_in_any_position():
    expected = [call("delete", targetId="abc")]
    actual = [call("get_recent"), call("search", q="bench"),
              call("delete", targetId="abc")]
    mode = cfg(allowed_extra_tools=["get_recent", "search"])
    assert score_tool_calls(expected, actual, mode).success
    # A non-allowed extra still fails.
    result = score_tool_calls(expected, actual + [call("rename")], mode)
    assert result.extra_tool_calls == ["rename"]


def test_allowed_extra_tools_on_no_tool_sample():
    mode = cfg(allowed_extra_tools=["get_recent"])
    assert score_tool_calls([], [call("get_recent")], mode).success
    result = score_tool_calls([], [call("get_recent"), call("log")], mode)
    assert not result.success
    assert result.extra_tool_calls == ["log"]


# --------------------------------------------------------- schema validation

SCHEMA = {
    "type": "object",
    "properties": {
        "workoutId": {"type": "string", "format": "uuid"},
        "reps": {"type": "integer", "minimum": 0},
    },
    "required": ["workoutId"],
    "additionalProperties": False,
}
UUID = "6f1c2a9e-3b7d-4c51-9a0e-2d8f4b6c1e37"


def test_schema_violation_fails_even_when_subset_matches():
    expected = [call("edit", workoutId=UUID)]
    actual = [call("edit", workoutId=UUID, reps=-1, junk=True)]
    mode = cfg(require_exact_arguments=False, validate_arguments_schema=True)
    result = score_tool_calls(expected, actual, mode,
                              tool_schemas={"edit": SCHEMA})
    assert not result.success
    paths = {e.argument_path for e in result.schema_errors}
    assert "reps" in paths and "" in paths  # bad value + additionalProperties
    assert classify_failure(result).value == "schema_violation"


def test_schema_checks_uuid_format():
    mode = cfg(require_exact_arguments=False, validate_arguments_schema=True)
    result = score_tool_calls([call("edit", workoutId={"$any": True})],
                              [call("edit", workoutId="workout-1")], mode,
                              tool_schemas={"edit": SCHEMA})
    assert not result.success
    assert "uuid" in result.schema_errors[0].message


def test_schema_validation_off_by_default():
    result = score_tool_calls([call("edit", workoutId={"$any": True})],
                              [call("edit", workoutId="workout-1")],
                              cfg(), tool_schemas={"edit": SCHEMA})
    assert result.success


def test_protocol_error_fails_run_even_with_no_expected_calls():
    result = score_tool_calls([], [], cfg(), protocol_error="invalid JSON envelope")
    assert not result.success
    assert classify_failure(result).value == "protocol_error"

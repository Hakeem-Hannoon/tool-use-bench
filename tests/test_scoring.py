"""Deterministic scorer behavior."""

from benchmark.schema import ScoringConfig, ToolCall
from benchmark.scoring import classify_failure, score_tool_calls


def call(name, /, **arguments):
    return ToolCall(name=name, arguments=arguments)


def cfg(**overrides):
    return ScoringConfig(**overrides)


def test_perfect_tool_call_passes():
    expected = [call("get_weather", location="Toronto, Canada", unit="celsius")]
    actual = [call("get_weather", location="Toronto, Canada", unit="celsius")]
    result = score_tool_calls(expected, actual, cfg())
    assert result.success
    assert result.reason == "Passed."
    assert classify_failure(result) is None


def test_missing_required_tool_fails():
    expected = [
        call("lookup_contact", name="Ana Silva"),
        call("send_slack_message", channel="#sales", message="hi"),
    ]
    actual = [call("lookup_contact", name="Ana Silva")]
    result = score_tool_calls(expected, actual, cfg(require_order=False))
    assert not result.success
    assert result.missing_tool_calls == ["send_slack_message"]
    assert classify_failure(result).value == "missing_tool_call"


def test_wrong_tool_name_fails():
    expected = [call("create_email_draft", to=["a@b.c"], subject="s", body="b")]
    actual = [call("send_email", to=["a@b.c"], subject="s", body="b")]
    result = score_tool_calls(expected, actual, cfg(require_order=False))
    assert not result.success
    assert result.wrong_tool_calls
    assert result.wrong_tool_calls[0].expected == "create_email_draft"
    assert result.wrong_tool_calls[0].actual == "send_email"
    assert classify_failure(result).value == "wrong_tool_name"


def test_wrong_argument_value_fails():
    expected = [call("get_weather", location="Toronto, Canada", unit="celsius")]
    actual = [call("get_weather", location="Toronto, Canada", unit="fahrenheit")]
    result = score_tool_calls(expected, actual, cfg())
    assert not result.success
    errors = result.argument_errors
    assert len(errors) == 1
    assert errors[0].argument_path == "unit"
    assert errors[0].error_type == "wrong_value"
    assert classify_failure(result).value == "wrong_argument"


def test_omitted_required_argument_fails():
    expected = [call("get_weather", location="Toronto, Canada", unit="celsius")]
    actual = [call("get_weather", location="Toronto, Canada")]
    result = score_tool_calls(expected, actual, cfg())
    assert not result.success
    assert any(e.error_type == "missing_argument" and e.argument_path == "unit"
               for e in result.argument_errors)


def test_extra_argument_fails_when_exact_required():
    expected = [call("search_calendar_events", query="dentist")]
    actual = [call("search_calendar_events", query="dentist",
                   start_date="2026-01-01")]
    result = score_tool_calls(expected, actual, cfg())
    assert not result.success
    assert any(e.error_type == "extra_argument" for e in result.argument_errors)


def test_extra_argument_allowed_when_not_exact():
    expected = [call("search_calendar_events", query="dentist")]
    actual = [call("search_calendar_events", query="dentist",
                   start_date="2026-01-01")]
    result = score_tool_calls(expected, actual,
                              cfg(require_exact_arguments=False))
    assert result.success


def test_extra_tool_call_fails_when_disallowed():
    expected = [call("get_weather", location="Berlin, Germany", unit="celsius")]
    actual = [
        call("get_weather", location="Berlin, Germany", unit="celsius"),
        call("get_weather", location="Paris, France", unit="celsius"),
    ]
    result = score_tool_calls(expected, actual, cfg(require_order=False))
    assert not result.success
    assert result.extra_tool_calls == ["get_weather"]
    assert classify_failure(result).value == "extra_tool_call"


def test_extra_tool_call_allowed_when_enabled():
    expected = [call("get_weather", location="Berlin, Germany", unit="celsius")]
    actual = [
        call("get_current_time", timezone="Europe/Berlin"),
        call("get_weather", location="Berlin, Germany", unit="celsius"),
    ]
    result = score_tool_calls(
        expected, actual,
        cfg(require_order=False, allow_extra_tool_calls=True))
    assert result.success


def test_required_order_enforced():
    expected = [
        call("lookup_contact", name="Ana Silva"),
        call("send_slack_message", channel="#sales", message="hi"),
    ]
    actual = list(reversed(expected))
    result = score_tool_calls(expected, actual, cfg(require_order=True))
    assert not result.success
    assert result.ordering_errors
    assert classify_failure(result).value == "ordering_error"


def test_order_ignored_when_configured():
    expected = [
        call("lookup_contact", name="Ana Silva"),
        call("send_slack_message", channel="#sales", message="hi"),
    ]
    actual = list(reversed(expected))
    result = score_tool_calls(expected, actual, cfg(require_order=False))
    assert result.success


def test_no_tool_sample_passes_without_calls():
    result = score_tool_calls([], [], cfg())
    assert result.success


def test_no_tool_sample_fails_with_any_call():
    result = score_tool_calls([], [call("web_search", query="x", num_results=1)],
                              cfg())
    assert not result.success
    assert result.extra_tool_calls == ["web_search"]
    assert classify_failure(result).value == "extra_tool_call"


def test_string_number_type_mismatch_fails_by_default():
    expected = [call("update_spreadsheet_cell", spreadsheet_id="S",
                     sheet_name="Q4", cell="C17", value="48250")]
    actual = [call("update_spreadsheet_cell", spreadsheet_id="S",
                   sheet_name="Q4", cell="C17", value=48250)]
    result = score_tool_calls(expected, actual, cfg())
    assert not result.success
    assert any(e.error_type == "type_mismatch" for e in result.argument_errors)


def test_coercion_bridges_string_and_number_when_enabled():
    expected = [call("convert_currency", amount=250, from_currency="USD",
                     to_currency="EUR")]
    actual = [call("convert_currency", amount="250", from_currency="USD",
                   to_currency="EUR")]
    strict = score_tool_calls(expected, actual, cfg())
    lenient = score_tool_calls(expected, actual,
                               cfg(allow_argument_coercion=True))
    assert not strict.success
    assert lenient.success


def test_int_and_float_are_both_json_numbers():
    expected = [call("convert_currency", amount=250, from_currency="USD",
                     to_currency="EUR")]
    actual = [call("convert_currency", amount=250.0, from_currency="USD",
                   to_currency="EUR")]
    assert score_tool_calls(expected, actual, cfg()).success


def test_whitespace_trimmed_only_when_enabled():
    expected = [call("lookup_contact", name="Priya Sharma")]
    actual = [call("lookup_contact", name="  Priya Sharma ")]
    trimmed = score_tool_calls(expected, actual, cfg())
    untrimmed = score_tool_calls(expected, actual,
                                 cfg(trim_string_whitespace=False))
    assert trimmed.success
    assert not untrimmed.success


def test_nested_object_arguments_compared_recursively():
    expected = [call("update_calendar_event", event_id="EVT-1",
                     changes={"date": "2026-08-07", "start_time": "10:00"})]
    actual = [call("update_calendar_event", event_id="EVT-1",
                   changes={"date": "2026-08-07", "start_time": "11:00"})]
    result = score_tool_calls(expected, actual, cfg())
    assert not result.success
    assert any(e.argument_path == "changes.start_time"
               for e in result.argument_errors)


def test_array_order_matters():
    expected = [call("create_github_issue", repo="a/b", title="t", body="b",
                     labels=["bug", "ios", "p1"])]
    actual = [call("create_github_issue", repo="a/b", title="t", body="b",
                   labels=["ios", "bug", "p1"])]
    assert not score_tool_calls(expected, actual, cfg()).success


def test_provider_error_and_timeout_fail_with_precedence():
    expected = [call("get_weather", location="X", unit="celsius")]
    errored = score_tool_calls(expected, [], cfg(), error="boom")
    assert not errored.success
    assert classify_failure(errored, error="boom").value == "provider_error"
    timed = score_tool_calls(expected, [], cfg(), timed_out=True)
    assert not timed.success
    assert classify_failure(timed, error="x", timed_out=True).value == "timeout"

"""The Spotter suite: structure, fixture grounding, and end-to-end episodes."""

import json
import re

import pytest

from fakes import ScriptedProvider, envelope, model_cfg, native, text

from benchmark.providers.base import ProviderSettings
from benchmark.runner import _record_from_result, run_benchmark
import benchmark.runner as runner_module
from benchmark.config import BenchmarkConfig
from benchmark.samples import load_benchmark_file
from benchmark.spotter_samples import (
    CATEGORIES, S_B1, S_B2, S_B3, W_YDAY, build_spotter_document,
    generate_spotter_sample_files, load_snapshot, validate_spotter_samples,
)

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@pytest.fixture(scope="module")
def snap():
    return load_snapshot()


@pytest.fixture(scope="module")
def samples(snap):
    return {s.id: s for s in validate_spotter_samples(snap)}


def test_suite_shape(samples):
    assert len(samples) == 70
    assert sum(s.category == "conversation" for s in samples.values()) == 15
    assert {s.category for s in samples.values()} == set(CATEGORIES)
    for s in samples.values():
        assert s.max_steps == 8
        assert s.scoring.validate_arguments_schema
        assert set(s.system_prompt_by_protocol) == {"spotter_router", "native"}


def test_each_sample_offers_exactly_its_route(snap, samples):
    for s in samples.values():
        route = s.notes.split("]")[0].strip("[").split()[0]
        assert [t.function.name for t in s.tools] == snap["domains"][route], s.id
        prompt = s.system_prompt_by_protocol["spotter_router"]
        assert prompt == snap["prompt_text"][f"{route}.spotter_router"]
        terminal = {t["name"] for t in snap["tools"] if t["terminal"]}
        assert set(s.terminal_tools) == terminal & s.tool_names()


def test_writes_are_never_free_extras(snap, samples):
    writes = {t["name"] for t in snap["tools"] if t["write"]}
    for s in samples.values():
        free = set(s.scoring.allowed_extra_tools) & writes
        assert free <= {"update_user_profile"}, (s.id, free)
        if free:
            assert s.category == "memory_and_privacy"


def test_every_pinned_id_comes_from_the_fixture_user(snap):
    doc = build_spotter_document(snap)
    world = json.dumps(doc["tool_result_sets"]["fixture_user"])
    for sample in doc["samples"]:
        pinned = json.dumps({
            "calls": sample["expected_tool_calls"],
            "state": sample.get("expected_state"),
        })
        for uuid in UUID_RE.findall(pinned):
            assert uuid in world, (sample["id"], uuid)


def test_generated_files_roundtrip(tmp_path, snap, samples):
    combined = generate_spotter_sample_files(tmp_path)
    loaded = {s.id: s for s in load_benchmark_file(combined)}
    assert loaded.keys() == samples.keys()
    per_file = list((tmp_path / "spotter").glob("*.json"))
    assert len(per_file) == len(samples)
    raw = json.loads(per_file[0].read_text())
    assert "tool_refs" in raw and "tools" not in raw  # resolved via the table


# ---------------------------------------------------------------- episodes


def _last_tool_result(conv) -> dict:
    content = conv.messages[-1]["content"]
    payload = content.split("\n", 1)[1].rsplit("\n\nContinue.", 1)[0]
    return json.loads(payload)


def router_cfg():
    return model_cfg(protocol="spotter_router")


async def test_router_edit_grounded_in_the_read_result(samples):
    sample = samples["0030_edit_bench_weight"]

    def script(conv, index):
        if index == 0:
            return envelope("search_workouts", exerciseName="bench")
        data = _last_tool_result(conv)
        workout = data["workouts"][0]
        bench = next(e for e in workout["exercises"] if e["name"] == "Bench Press")
        return envelope("create_pending_workout_edit", workoutId=workout["workoutId"],
                        summary="Bench 185 → 195",
                        setUpdates=[{"setId": st["setId"], "weight": 195}
                                    for st in reversed(bench["sets"])])

    provider = ScriptedProvider(router_cfg(), ProviderSettings(), script)
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(sample, result, save_raw=False, model_cfg=router_cfg())
    assert record.success, record.scoring_result.reason
    assert provider.requests[0]["system"].startswith("You are Spotter")
    assert provider.requests[0]["tools"] is None


async def test_router_edit_with_invented_ids_fails(samples):
    sample = samples["0030_edit_bench_weight"]
    fake = "11111111-2222-4333-8444-555555555555"
    provider = ScriptedProvider(router_cfg(), ProviderSettings(), [
        envelope("create_pending_workout_edit", workoutId=fake, summary="x",
                 setUpdates=[{"setId": fake, "weight": 195}]),
    ])
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(sample, result, save_raw=False, model_cfg=router_cfg())
    assert not record.success
    assert record.failure_type == "wrong_argument"


async def test_router_schema_violation_is_caught(samples):
    sample = samples["0070_rest_three_minutes"]
    provider = ScriptedProvider(router_cfg(), ProviderSettings(), [
        envelope("control_rest_timer", action="start", seconds=180, sound="loud"),
    ])
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(sample, result, save_raw=False, model_cfg=router_cfg())
    assert record.failure_type == "schema_violation"


async def test_native_decoy_with_a_harmless_read_passes(samples):
    sample = samples["0100_bare_load"]
    provider = ScriptedProvider(model_cfg(), ProviderSettings(), [
        native(("get_recent_workouts", {"limit": 3})),
        text("Which lift was that — bench, squat, deadlift?"),
    ])
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(sample, result, save_raw=False, model_cfg=model_cfg())
    assert record.success, record.scoring_result.reason
    assert provider.requests[0]["system"] == sample.system_prompt_by_protocol["native"]
    assert "create_pending_workout_session" in provider.requests[0]["tools"]


async def test_native_decoy_that_guesses_a_lift_fails(samples):
    sample = samples["0100_bare_load"]
    provider = ScriptedProvider(model_cfg(), ProviderSettings(), [
        native(("create_pending_workout_session",
                {"exercises": [{"name": "Squat", "sets": [{"weight": 225, "reps": 5}]}]})),
    ])
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(sample, result, save_raw=False, model_cfg=model_cfg())
    assert not record.success and record.failure_type == "extra_tool_call"


async def test_branded_lookup_requires_grounded_numbers(samples):
    sample = samples["0053_branded_lookup"]

    def script(conv, index):
        if index == 0:
            return envelope("lookup_nutrition", query="Quest cookies and cream bar")
        food = _last_tool_result(conv)["results"][0]
        return envelope("create_pending_macro_log", mealType="snack", items=[{
            "description": food["description"], "proteinG": food["proteinG"],
            "carbsG": food["carbsG"], "fatG": food["fatG"],
            "calories": food["calories"], "isEstimate": False,
            "foodId": food["foodId"]}])

    provider = ScriptedProvider(router_cfg(), ProviderSettings(), script)
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(sample, result, save_raw=False, model_cfg=router_cfg())
    assert record.success, record.scoring_result.reason


async def test_runner_end_to_end_on_the_spotter_file(tmp_path, monkeypatch, samples):
    combined = generate_spotter_sample_files(tmp_path / "bench")
    data = json.loads(combined.read_text())
    keep = {"0030_edit_bench_weight", "0070_rest_three_minutes"}
    data["samples"] = [s for s in data["samples"] if s["id"] in keep]
    data["sample_count"] = len(data["samples"])
    combined.write_text(json.dumps(data))

    def make(cfg, settings):
        def script(conv, index):
            prompt = conv.messages[0]["content"]
            if "timer" in prompt or "rest" in prompt:
                return envelope("control_rest_timer", action="start", seconds=180)
            return envelope("create_pending_workout_edit", workoutId=W_YDAY,
                            summary="bench 195", setUpdates=[
                                {"setId": i, "weight": 195} for i in (S_B1, S_B2, S_B3)])
        return ScriptedProvider(cfg, settings, script)

    monkeypatch.setattr(runner_module, "create_provider", make)
    config = BenchmarkConfig.model_validate({
        "benchmark": str(combined),
        "output": str(tmp_path / "out" / "results.json"),
        "report": str(tmp_path / "out" / "report.md"),
        "runs_per_sample": 2, "max_retries": 0, "timeout_seconds": 10,
        "models": [{"name": "grok-4-1-fast", "provider": "openai_compatible",
                    "endpoint": "https://example.invalid/openai/v1",
                    "protocol": "spotter_router",
                    "pricing": {"input_per_mtok": 0.2, "output_per_mtok": 0.5,
                                "cached_input_per_mtok": 0.05}}],
    })
    results = await run_benchmark("config.spotter.json", config, resume=True, force=False)
    (summary,) = results["model_summaries"]
    assert summary["success_rate"] == 1.0
    assert summary["protocol"] == "spotter_router"
    assert summary["cost_per_run_usd"] is not None
    report = (tmp_path / "out" / "report.md").read_text()
    assert "## Workload Picks" in report and "id_grounded_edits" in report


# ------------------------------------------------------------------ oracle
#
# Materialize every expectation into a concrete call (the answer a perfect
# model would give) and check that it scores as a pass and that none of its
# pinned values would be rejected by the Zod-derived schema. A sample that
# fails here could never be passed by any model.

REGEX_EXAMPLES = ["Bench Press", "Squat", "Deadlift", "Incline DB Press",
                  "Treadmill", "StairMaster", "Stationary Bike", "Romanian Deadlift",
                  "Face Pulls", "Pull", "Barbell Row", "2026-10-21", "2026-10-01",
                  "2026-10-08", "2026-10-15"]


def materialize(value):
    from benchmark.scoring import is_matcher
    if is_matcher(value):
        op, arg = next(iter(value.items()))
        if op == "$any":
            return "anything"
        if op == "$ci":
            return arg
        if op == "$contains":
            return " ".join([arg] if isinstance(arg, str) else arg)
        if op == "$regex":
            return next(x for x in REGEX_EXAMPLES if re.fullmatch(arg, x))
        if op == "$range":
            lo, hi = arg
            return lo if hi is None else hi if lo is None else (lo + hi) / 2
        if op == "$one_of":
            return materialize(arg[0])
        if op in ("$unordered", "$includes"):
            return [materialize(a) for a in arg]
        if op == "$optional":
            return materialize(arg)
        raise AssertionError(op)
    if isinstance(value, dict):
        return {k: materialize(v) for k, v in value.items()
                if not (isinstance(v, dict) and set(v) & {"$absent", "$optional"})}
    if isinstance(value, list):
        return [materialize(v) for v in value]
    return value


def test_every_expectation_is_satisfiable_and_schema_valid(samples):
    from benchmark.schema import ToolCall
    from benchmark.scoring import schema_errors_for, score_tool_calls
    for s in samples.values():
        calls = [ToolCall(name=c.name, arguments=materialize(c.arguments))
                 for c in s.expected_tool_calls]
        result = score_tool_calls(s.expected_tool_calls, calls,
                                  s.scoring.model_copy(update={
                                      "validate_arguments_schema": False}))
        assert result.success, (s.id, result.reason)
        schemas = s.tool_schemas()
        for c in calls:
            bad = [e for e in schema_errors_for(c, schemas[c.name])
                   if "is a required property" not in e.message
                   and not e.message.startswith("'anything'")]
            assert not bad, (s.id, [(e.argument_path, e.message) for e in bad])

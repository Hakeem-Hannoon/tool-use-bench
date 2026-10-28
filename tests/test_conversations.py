"""Multi-turn Spotter dialogues scored on the final world, not the trace."""

import json

import pytest

from fakes import ScriptedProvider, envelope, model_cfg, text

from benchmark.providers.base import ProviderSettings
from benchmark.runner import _record_from_result
from benchmark.spotter_samples import (
    S_B1, S_B2, S_B3, W_PULL, W_PUSH, W_YDAY, build_spotter_document,
)
from benchmark.world import World, canonical_movement, diff_world
from benchmark.schema import ExpectedState


def _doc():
    return build_spotter_document()


def _world():
    return World.from_tool_results(_doc()["tool_result_sets"]["fixture_user"])


def _payload(content: str) -> dict:
    body = content.split("\n", 1)[1].rsplit("\n\nContinue.", 1)[0]
    return json.loads(body)


def test_aliases_fold_bench_and_squat():
    assert canonical_movement("Bench Press") == canonical_movement("barbell bench")
    assert canonical_movement("Squats") == canonical_movement("Back Squat")
    assert canonical_movement("Leg Press") != canonical_movement("Squat")
    assert canonical_movement("Romanian Deadlift") != canonical_movement("Deadlift")


def test_create_is_visible_to_the_next_read():
    from benchmark.schema import ToolCall

    world = _world()
    result = world.apply(ToolCall(
        name="create_pending_workout_session",
        arguments={"exercises": [{"name": "Bench Press", "sets": [
            {"weight": 185, "reps": 5}, {"weight": 185, "reps": 5},
            {"weight": 185, "reps": 5}]}]}))
    assert result.ends_turn
    recent = world.apply(ToolCall(name="get_recent_workouts",
                                  arguments={"limit": 1}))
    workout = recent.payload["workouts"][0]
    assert workout["workoutId"] != W_YDAY
    assert workout["date"] == world.today
    assert [s["weight"] for s in workout["exercises"][0]["sets"]] == [185, 185, 185]


def test_same_day_bench_does_not_double_log():
    from benchmark.schema import ToolCall

    world = _world()
    world.apply(ToolCall(
        name="create_pending_workout_session",
        arguments={"exercises": [{"name": "Bench Press", "sets": [
            {"weight": 185, "reps": 5}]}]}))
    again = world.apply(ToolCall(
        name="create_pending_workout_session",
        arguments={"exercises": [{"name": "bench", "sets": [
            {"weight": 195, "reps": 5}]}]}))
    assert not again.ends_turn
    assert again.payload["status"] == "same_day_overlap"
    created = [w for w in world.project()["workouts"] if w["origin"] == "created"]
    assert len(created) == 1
    assert created[0]["exercises"][0]["sets"][0]["weight"] == 185


def test_unknown_set_id_does_not_commit():
    from benchmark.schema import ToolCall

    world = _world()
    before = world.project()
    result = world.apply(ToolCall(
        name="create_pending_workout_edit",
        arguments={"workoutId": W_YDAY, "summary": "nope",
                   "setUpdates": [{"setId": "11111111-2222-4333-8444-555555555555",
                                   "weight": 999}]}))
    assert not result.ends_turn
    assert world.project()["workouts"] == before["workouts"]


def test_delete_and_memory_round_trip():
    from benchmark.schema import ToolCall

    world = _world()
    world.apply(ToolCall(name="remember",
                         arguments={"content": "left knee hurts on deep squats",
                                    "kind": "health"}))
    world.apply(ToolCall(name="forget", arguments={"content": "knee"}))
    world.apply(ToolCall(name="remember",
                         arguments={"content": "right shoulder is cranky",
                                    "kind": "health"}))
    world.apply(ToolCall(name="create_pending_workout_delete",
                         arguments={"target": "set", "targetId": S_B2,
                                    "summary": "drop set 2"}))
    state = world.project()
    assert state["memories"] == [{"content": "right shoulder is cranky",
                                  "kind": "health"}]
    assert S_B2 in state["deleted_ids"]
    yesterday = next(w for w in state["workouts"] if w["workoutId"] == W_YDAY)
    bench = yesterday["exercises"][0]
    assert [s["setId"] for s in bench["sets"]] == [S_B1, S_B3]


def test_diff_accepts_either_grouping_of_two_lifts():
    world = _world()
    initial = world.project()
    from benchmark.schema import ToolCall

    world.apply(ToolCall(
        name="create_pending_workout_session",
        arguments={"exercises": [
            {"name": "Bench Press", "sets": [{"weight": 135, "reps": 8}]},
            {"name": "Squat", "sets": [{"weight": 235, "reps": 5}]}]}))
    together = world.project()
    split_world = _world()
    split_world.apply(ToolCall(
        name="create_pending_workout_session",
        arguments={"exercises": [
            {"name": "Bench Press", "sets": [{"weight": 135, "reps": 8}]}]}))
    split_world.apply(ToolCall(
        name="create_pending_workout_session",
        arguments={"date": split_world.today, "exercises": [
            {"name": "Back Squat", "sets": [{"weight": 235, "reps": 5}]}]}))
    expected = ExpectedState.model_validate({
        "unchanged_workouts": [W_YDAY, W_PULL, W_PUSH],
        "created_exercises": [
            {"date": world.today, "name": {"$regex": "(?i).*(bench).*"},
             "sets": [{"setNumber": 1, "weight": 135, "reps": 8,
                       "weightUnit": "lb"}]},
            {"date": world.today, "name": {"$regex": "(?i).*(squat).*"},
             "sets": [{"setNumber": 1, "weight": 235, "reps": 5,
                       "weightUnit": "lb"}]},
        ],
    })
    assert diff_world(expected, together, initial) == []
    assert diff_world(expected, split_world.project(), initial) == []


# ---------------------------------------------------------------- episodes


def _samples():
    from benchmark.spotter_samples import validate_spotter_samples
    return {s.id: s for s in validate_spotter_samples()}


def _bench_sets(workout: dict, weight: int) -> list[dict]:
    bench = next(ex for ex in workout["exercises"] if "bench" in ex["name"].lower())
    return [{"setId": s["setId"], "weight": weight} for s in bench["sets"]]


def _new_workout(payload: dict) -> dict:
    return next(w for w in payload["workouts"] if w["workoutId"] not in
                {W_YDAY, W_PULL, W_PUSH})


@pytest.fixture(scope="module")
def samples():
    return _samples()


def _script_raise_all(read_tool: str):
    def script(conv, index):
        last = conv.messages[-1]["content"]
        if isinstance(last, str) and last.startswith("Make all three"):
            if read_tool == "search_workouts":
                return envelope("search_workouts", exerciseName="bench")
            return envelope("get_recent_workouts", limit=5)
        if isinstance(last, str) and last.startswith("TOOL_RESULT"):
            workout = _new_workout(_payload(last))
            return envelope("create_pending_workout_edit",
                            workoutId=workout["workoutId"],
                            summary="bench to 195",
                            setUpdates=_bench_sets(workout, 195))
        if isinstance(last, str) and last.startswith("Leave yesterday"):
            return envelope(reply="Yesterday is untouched.")
        return envelope(
            "create_pending_workout_session",
            exercises=[{"name": "Bench Press", "sets": [
                {"weight": 185, "reps": 5}, {"weight": 185, "reps": 5},
                {"weight": 185, "reps": 5}]}])
    return script


async def test_search_and_recent_reads_both_pass(samples):
    sample = samples["0200_log_then_raise_all"]
    for read_tool in ("search_workouts", "get_recent_workouts"):
        provider = ScriptedProvider(
            model_cfg(protocol="spotter_router"), ProviderSettings(),
            _script_raise_all(read_tool))
        result = await provider.complete(sample, 0, "")
        record = _record_from_result(
            sample, result, save_raw=False,
            model_cfg=model_cfg(protocol="spotter_router"))
        assert record.success, (read_tool, record.scoring_result.reason)
        assert any(c.name == read_tool for c in record.actual_tool_calls)
        assert record.world_state is not None
        assert record.expected_state is not None


async def test_editing_yesterday_instead_fails(samples):
    sample = samples["0200_log_then_raise_all"]

    def script(conv, index):
        last = conv.messages[-1]["content"]
        if isinstance(last, str) and last.startswith("Make all three"):
            return envelope("create_pending_workout_edit", workoutId=W_YDAY,
                            summary="yesterday bench",
                            setUpdates=[{"setId": i, "weight": 195}
                                        for i in (S_B1, S_B2, S_B3)])
        if isinstance(last, str) and last.startswith("Leave yesterday"):
            return envelope(reply="Done.")
        return envelope(
            "create_pending_workout_session",
            exercises=[{"name": "Bench Press", "sets": [
                {"weight": 185, "reps": 5}] * 3}])

    provider = ScriptedProvider(
        model_cfg(protocol="spotter_router"), ProviderSettings(), script)
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(
        sample, result, save_raw=False,
        model_cfg=model_cfg(protocol="spotter_router"))
    assert not record.success
    assert record.failure_type == "world_mismatch"
    assert record.actual_tool_calls  # the trace is still on the run


async def test_invalid_create_can_be_retried_in_the_turn(samples):
    sample = samples["0202_log_then_delete_it"]

    def script(conv, index):
        last = conv.messages[-1]["content"]
        if isinstance(last, str) and "invalid_arguments" in last:
            return envelope(
                "create_pending_workout_session",
                exercises=[{"name": "Bench Press", "sets": [
                    {"weight": 185, "reps": 5}] * 3}])
        if isinstance(last, str) and last.startswith("Delete that"):
            return envelope("get_recent_workouts", limit=1)
        if isinstance(last, str) and "get_recent_workouts" in last:
            workout = _payload(last)["workouts"][0]
            return envelope("create_pending_workout_delete", target="workout",
                            targetId=workout["workoutId"], summary="remove it")
        if isinstance(last, str) and last.startswith("Yesterday"):
            return envelope(reply="Left it.")
        return envelope("create_pending_workout_session", exercises=[])

    provider = ScriptedProvider(
        model_cfg(protocol="spotter_router"), ProviderSettings(), script)
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(
        sample, result, save_raw=False,
        model_cfg=model_cfg(protocol="spotter_router"))
    assert record.success, record.scoring_result.reason


async def test_native_dialogue_commits_across_turns(samples):
    from fakes import native

    sample = samples["0200_log_then_raise_all"]

    def script(conv, index):
        last = conv.messages[-1]
        content = last.get("content")
        if isinstance(content, str) and content.startswith("Make all three"):
            return native(("get_recent_workouts", {"limit": 5}))
        if last.get("role") == "tool" and last.get("name") == "get_recent_workouts":
            workout = _new_workout(json.loads(content))
            return native(("create_pending_workout_edit", {
                "workoutId": workout["workoutId"],
                "summary": "bench to 195",
                "setUpdates": _bench_sets(workout, 195),
            }))
        if isinstance(content, str) and content.startswith("Leave yesterday"):
            return text("Yesterday is untouched.")
        return native(("create_pending_workout_session", {
            "exercises": [{"name": "Bench Press", "sets": [
                {"weight": 185, "reps": 5}] * 3}]}))

    provider = ScriptedProvider(model_cfg(), ProviderSettings(), script)
    result = await provider.complete(sample, 0, "")
    record = _record_from_result(sample, result, save_raw=False,
                                 model_cfg=model_cfg())
    assert record.success, record.scoring_result.reason
    assert {s.turn for s in result.steps} == {1, 2, 3}

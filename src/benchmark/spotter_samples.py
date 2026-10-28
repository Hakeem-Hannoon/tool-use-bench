"""The Spotter suite: Spotter's real tools, prompts, and routing.

Every sample is built from the snapshot exported by
``spotter/sync_from_gym_bro.mjs``: the tools a sample offers are exactly the
tools Spotter's deterministic domain router (general / nutrition /
compound) exposes for that message, with the JSON Schema of the Zod
validator production parses them with; the system prompt is
``buildSystemPrompt`` output for that route, in router or native tool mode.

Samples run as episodes of up to ``MAX_STEPS`` (production's router cap).
Read tools answer from one consistent fixture user — "Alex", lb, a
push/pull/legs plan, yesterday's lower + bench session — whose database ids
are fixed UUIDs, so an edit or delete only scores when it targets the exact
id a read returned. Writes and display cards end the episode, as in
production.

Scoring pins what Spotter needs: tool choice, the numbers, units, dates and
ids, and argument validity against the Zod-derived schema. Free text (intro,
summary, captions, titles) is never pinned, and harmless reads before an
answer are always allowed. Reply text is not scored.

The ``conversation`` category is the exception: each sample is 3–5 user
turns against this same fixture, and the run passes when the final world
matches ``expected_state``. The calls are stored on the run and not scored,
so a read via ``search_workouts`` and a read via ``get_recent_workouts``
both pass when they leave the same sets, deletes, and memories.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from . import BENCHMARK_NAME, __version__
from .samples import SampleValidationError, validate_samples
from .utils.json_utils import atomic_write_json, load_json_file

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "spotter" / "snapshot"
SUITE_DIR = "spotter"
COMBINED_FILENAME = "spotter_benchmark.json"
#: Production's router loop cap (MAX_STEPS in orchestrator.ts).
MAX_STEPS = 8

CATEGORIES = (
    "workout_logging",
    "history_reads",
    "id_grounded_edits",
    "charts",
    "nutrition",
    "body_metrics",
    "device_actions",
    "plans",
    "settings",
    "clarify_or_decline",
    "memory_and_privacy",
    "compound",
    "conversation",
)

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def load_snapshot(snapshot_dir: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    path = snapshot_dir / "tools.json"
    if not path.exists():
        raise SampleValidationError(
            f"Spotter snapshot not found at {path}. Run "
            f"`node spotter/sync_from_gym_bro.mjs <path-to-gym-bro>` first.")
    snap = load_json_file(path)
    snap["prompt_text"] = {
        pid: (snapshot_dir / rel).read_text(encoding="utf-8")
        for pid, rel in snap["prompts"].items()
    }
    snap["by_name"] = {t["name"]: t for t in snap["tools"]}
    return snap


# ---------------------------------------------------------------------------
# Matcher shorthands
# ---------------------------------------------------------------------------

ANY = {"$any": True}
ABSENT = {"$absent": True}


def OPT(value: Any) -> dict[str, Any]:  # noqa: N802 - reads like a matcher
    return {"$optional": value}


def CI(text: str) -> dict[str, Any]:  # noqa: N802
    return {"$ci": text}


def RX(pattern: str) -> dict[str, Any]:  # noqa: N802
    return {"$regex": pattern}


def HAS(*fragments: str) -> dict[str, Any]:  # noqa: N802
    return {"$contains": list(fragments) if len(fragments) > 1 else fragments[0]}


def RANGE(lo: float | None, hi: float | None) -> dict[str, Any]:  # noqa: N802
    return {"$range": [lo, hi]}


def ONE_OF(*values: Any) -> dict[str, Any]:  # noqa: N802
    return {"$one_of": list(values)}


def UNORDERED(*items: Any) -> dict[str, Any]:  # noqa: N802
    return {"$unordered": list(items)}


def INCLUDES(*items: Any) -> dict[str, Any]:  # noqa: N802
    return {"$includes": list(items)}


BENCH = RX(r"(?i)(flat |barbell )*bench( press)?( \(barbell\))?")
SQUAT = RX(r"(?i)(barbell |back )*squats?( \(barbell\))?")
DEADLIFT = RX(r"(?i)(conventional |barbell )*deadlifts?( \(barbell\))?")
INCLINE_DB = RX(r"(?i)incline (db|dumbbell)s?( bench)?( press)?")


def S(weight: float | None, reps: int, unit: str = "lb", *,  # noqa: N802
      set_type: Any = None) -> dict[str, Any]:
    """One logged set: weight, reps, unit (optional when it is the default lb),
    and setType (optional 'working' unless given)."""
    out: dict[str, Any] = {"reps": reps}
    if weight is not None:
        out["weight"] = weight
        out["weightUnit"] = OPT(unit) if unit == "lb" else unit
    out["setType"] = set_type if set_type is not None else OPT("working")
    return out


def call(tool: str, /, alternatives: list[dict[str, Any]] | None = None,
         **arguments: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"name": tool, "arguments": arguments}
    if alternatives:
        out["alternatives"] = alternatives
    return out


def alt(tool: str, /, **arguments: Any) -> dict[str, Any]:
    return {"name": tool, "arguments": arguments}


# ---------------------------------------------------------------------------
# The fixture user
# ---------------------------------------------------------------------------

W_YDAY = "3f8a6c2e-1b4d-4e7a-9c5f-6d2b8e1a4c70"
EX_BENCH = "a1c4e7b2-5d8f-4a1c-8e3b-7f2d6a9c1e45"
S_B1 = "b2d5f8c3-6e9a-4b2d-9f4c-8a3e7b1d2f56"
S_B2 = "c3e6a9d4-7f1b-4c3e-8a5d-9b4f8c2e3a67"
S_B3 = "d4f7b1e5-8a2c-4d4f-9b6e-1c5a9d3f4b78"
EX_SQUAT = "e5a8c2f6-9b3d-4e5a-8c7f-2d6b1e4a5c89"
S_Q1 = "f6b9d3a7-1c4e-4f6b-9d8a-3e7c2f5b6d91"
S_Q2 = "a7c1e4b8-2d5f-4a7c-8e9b-4f8d3a6c7e12"
S_Q3 = "b8d2f5c9-3e6a-4b8d-9f1c-5a9e4b7d8f23"
W_PULL = "5c2e9a7d-4f1b-4c8e-a3d6-8b5f2e9c1a34"
EX_ROW = "6d3f1b8e-5a2c-4d9f-b4e7-9c6a3f1d2b45"
EX_PULLUP = "7e4a2c9f-6b3d-4e1a-85f8-1d7b4a2e3c56"
W_PUSH = "8f5b3d1a-7c4e-4f2b-96a9-2e8c5b3f4d67"
EX_BENCH_OLD = "9a6c4e2b-8d5f-4a3c-a7b1-3f9d6c4a5e78"
EX_OHP = "1b7d5f3c-9e6a-4b4d-b8c2-4a1e7d5b6f89"
PLAN_ID = "2c8e6a4d-1f7b-4c5e-89d3-5b2f8e6c7a91"
SM_BREAKFAST = "4e1a8c6f-3b9d-4e7a-8b5f-7d4b1a8e9c13"
SM_SHAKE = "5f2b9d7a-4c1e-4f8b-9c6a-8e5c2b9f1d24"
MEAL_PLAN_ID = "6a3c1e8b-5d2f-4a9c-ad7b-9f6d3c1a2e35"
FOOD_QUEST = "7b4d2f9c-6e3a-4b1d-be8c-1a7e4d2b3f46"
FOOD_EGG = "8c5e3a1d-7f4b-4c2e-8f9d-2b8f5e3c4a57"
FOOD_SHAKE = "9d6f4b2e-8a5c-4d3f-9a1e-3c9a6f4d5b68"
BARCODE = "0850006801029"


def _sets(ids: list[str], weight: float | None, reps: list[int],
          e1rm: float | None) -> list[dict[str, Any]]:
    return [{"setId": sid, "setNumber": i + 1, "weight": weight,
             "weightUnit": "lb" if weight is not None else None, "reps": r,
             "durationSeconds": None, "distance": None, "distanceUnit": None,
             "speed": None, "calories": None, "rpe": None, "setType": "working",
             "estimated1rm": e1rm}
            for i, (sid, r) in enumerate(zip(ids, reps))]


def _world(today: str) -> list[dict[str, Any]]:
    """Scripted read-tool results for the fixture user, most specific first."""
    d = date.fromisoformat(today)

    def ago(days: int) -> str:
        return (d - timedelta(days=days)).isoformat()

    yday = {
        "workoutId": W_YDAY, "date": ago(1), "title": "Lower + bench",
        "exercises": [
            {"workoutExerciseId": EX_BENCH, "name": "Bench Press",
             "sets": _sets([S_B1, S_B2, S_B3], 185, [5, 5, 5], 215.8)},
            {"workoutExerciseId": EX_SQUAT, "name": "Squat",
             "sets": _sets([S_Q1, S_Q2, S_Q3], 315, [5, 5, 5], 367.5)},
        ],
    }
    pull = {
        "workoutId": W_PULL, "date": ago(4), "title": "Pull",
        "exercises": [
            {"workoutExerciseId": EX_ROW, "name": "Barbell Row",
             "sets": _sets(["0c1a2b3c-4d5e-4f60-8172-839405a6b7c8",
                            "1d2b3c4d-5e6f-4071-9283-94a516b7c8d9",
                            "2e3c4d5e-6f70-4182-a394-a5b627c8d9ea"], 155, [8, 8, 8], 196.3)},
            {"workoutExerciseId": EX_PULLUP, "name": "Pull-Up",
             "sets": _sets(["3f4d5e6f-7081-4293-b4a5-b6c738d9eafb",
                            "40516273-8495-4a6b-8c7d-8e9fa0b1c2d3",
                            "51627384-95a6-4b7c-9d8e-9fa0b1c2d3e4"], None, [10, 8, 7], None)},
        ],
    }
    push = {
        "workoutId": W_PUSH, "date": ago(7), "title": "Push",
        "exercises": [
            {"workoutExerciseId": EX_BENCH_OLD, "name": "Bench Press",
             "sets": _sets(["62738495-a6b7-4c8d-8e9f-a0b1c2d3e4f5",
                            "738495a6-b7c8-4d9e-9fa0-b1c2d3e4f506",
                            "8495a6b7-c8d9-4eaf-a0b1-c2d3e4f50617"], 180, [5, 5, 5], 210.0)},
            {"workoutExerciseId": EX_OHP, "name": "Overhead Press",
             "sets": _sets(["95a6b7c8-d9ea-4fb0-b1c2-d3e4f5061728",
                            "a6b7c8d9-eafb-4c01-82d3-e4f506172839",
                            "b7c8d9ea-fb0c-4d12-93e4-f5061728394a"], 115, [6, 6, 6], 138.0)},
        ],
    }
    bench_trend = [{"date": ago(43), "estimated1rm": 205},
                   {"date": ago(29), "estimated1rm": 210},
                   {"date": ago(15), "estimated1rm": 213},
                   {"date": ago(1), "estimated1rm": 216}]
    weights = [{"date": ago(28), "weight": 186.2}, {"date": ago(21), "weight": 185.4},
               {"date": ago(14), "weight": 184.6}, {"date": ago(7), "weight": 183.9},
               {"date": today, "weight": 183.0}]
    plan_slots = [
        {"position": 0, "weekday": "Monday", "label": "Push", "isRest": False,
         "exercises": [{"name": "Bench Press", "sets": 4, "reps": "5", "note": None},
                       {"name": "Overhead Press", "sets": 3, "reps": "6-8", "note": None}]},
        {"position": 1, "weekday": "Tuesday", "label": "Rest", "isRest": True, "exercises": []},
        {"position": 2, "weekday": "Wednesday", "label": "Pull", "isRest": False,
         "exercises": [{"name": "Barbell Row", "sets": 3, "reps": "8", "note": None},
                       {"name": "Pull-Up", "sets": 3, "reps": "AMRAP", "note": None}]},
        {"position": 3, "weekday": "Thursday", "label": "Rest", "isRest": True, "exercises": []},
        {"position": 4, "weekday": "Friday", "label": "Legs", "isRest": False,
         "exercises": [{"name": "Squat", "sets": 3, "reps": "5", "note": None},
                       {"name": "Romanian Deadlift", "sets": 3, "reps": "8", "note": None}]},
        {"position": 5, "weekday": "Saturday", "label": "Rest", "isRest": True, "exercises": []},
        {"position": 6, "weekday": "Sunday", "label": "Rest", "isRest": True, "exercises": []},
    ]
    none_found = {"weightUnit": "lb", "totalMatches": 0, "returned": 0, "workouts": []}
    return [
        # --- workouts -----------------------------------------------------
        {"tool": "get_recent_workouts", "when": {"limit": RANGE(None, 1)},
         "result": {"weightUnit": "lb", "count": 1, "workouts": [yday]}},
        {"tool": "get_recent_workouts",
         "result": {"weightUnit": "lb", "count": 3, "workouts": [yday, pull, push]}},
        {"tool": "search_workouts", "when": {"exerciseName": RX(r"(?i).*leg ?press.*")},
         "result": none_found},
        {"tool": "search_workouts", "when": {"query": RX(r"(?i).*leg ?press.*")},
         "result": none_found},
        {"tool": "search_workouts", "when": {"exerciseName": RX(r"(?i).*bench.*")},
         "result": {"weightUnit": "lb", "totalMatches": 2, "returned": 2,
                    "workouts": [yday, push]}},
        {"tool": "search_workouts", "when": {"exerciseName": RX(r"(?i).*squat.*")},
         "result": {"weightUnit": "lb", "totalMatches": 1, "returned": 1, "workouts": [yday]}},
        {"tool": "search_workouts",
         "result": {"weightUnit": "lb", "totalMatches": 3, "returned": 3,
                    "workouts": [yday, pull, push]}},
        {"tool": "get_exercise_history", "when": {"exerciseName": RX(r"(?i).*bench.*")},
         "result": {"found": True, "weightUnit": "lb", "history": {
             "exerciseId": "c8d9eafb-0c1d-4e23-a4f5-061728394a5b",
             "exerciseName": "Bench Press", "weightUnit": "lb",
             "bestSet": {"weight": 185, "reps": 5, "estimated1rm": 216, "date": ago(1)},
             "estimated1rmTrend": bench_trend,
             "volumeTrend": [{"date": t["date"], "volume": v}
                             for t, v in zip(bench_trend, [2550, 2600, 2700, 2775])],
             "recentSessions": [
                 {"workoutId": W_YDAY, "date": ago(1),
                  "sets": [{"weight": 185, "reps": 5, "durationSeconds": None,
                            "estimated1rm": 215.8}] * 3},
                 {"workoutId": W_PUSH, "date": ago(7),
                  "sets": [{"weight": 180, "reps": 5, "durationSeconds": None,
                            "estimated1rm": 210.0}] * 3}]}}},
        {"tool": "get_exercise_history", "when": {"exerciseName": RX(r"(?i).*row.*")},
         "result": {"found": True, "weightUnit": "lb", "history": {
             "exerciseId": "d9eafb0c-1d2e-4f34-b506-1728394a5b6c",
             "exerciseName": "Barbell Row", "weightUnit": "lb",
             "bestSet": {"weight": 155, "reps": 8, "estimated1rm": 196, "date": ago(4)},
             "estimated1rmTrend": [{"date": ago(4), "estimated1rm": 196}],
             "volumeTrend": [{"date": ago(4), "volume": 3720}],
             "recentSessions": [{"workoutId": W_PULL, "date": ago(4),
                                 "sets": [{"weight": 155, "reps": 8,
                                           "durationSeconds": None,
                                           "estimated1rm": 196.3}] * 3}]}}},
        {"tool": "get_exercise_history",
         "result": {"found": False, "query": "unknown"}},
        {"tool": "get_progress_summary",
         "result": {"weightUnit": "lb", "streakDays": 3, "workoutsThisWeek": 2,
                    "weeklyVolume": [{"weekStart": ago(9), "volume": 18450},
                                     {"weekStart": ago(2), "volume": 12900}],
                    "recentPRs": [{"exercise": "Bench Press", "weight": 185, "reps": 5,
                                   "estimated1rm": 216, "date": ago(1)}]}},
        {"tool": "get_bodyweight_history",
         "result": {"weightUnit": "lb", "count": len(weights), "points": weights}},
        # --- plan -----------------------------------------------------------
        {"tool": "get_workout_plan",
         "result": {"hasPlan": True, "planId": PLAN_ID, "name": "PPL", "planType": "weekly",
                    "slots": plan_slots}},
        {"tool": "get_progression_suggestions",
         "result": {"hasPlan": True, "weekday": "Wednesday", "label": "Pull", "isRest": False,
                    "exercises": [
                        {"name": "Barbell Row", "sets": 3, "reps": "8",
                         "suggestion": {"kind": "increase", "weight": 160, "reps": 8,
                                        "reason": "Hit 3x8 at 155 last session."}},
                        {"name": "Pull-Up", "sets": 3, "reps": "AMRAP",
                         "suggestion": {"kind": "hold", "weight": None, "reps": None,
                                        "reason": "Bodyweight; beat 10/8/7."}}]}},
        # --- profile & memory -------------------------------------------------
        {"tool": "get_user_profile",
         "result": {"displayName": "Alex", "preferredWeightUnit": "lb",
                    "preferredDistanceUnit": "mi", "trainingGoal": "get stronger",
                    "experienceLevel": "intermediate", "injuries": None,
                    "equipmentAccess": "full gym", "humourLevel": 5,
                    "responseLength": "normal", "confirmBeforeLogging": True,
                    "coachPersona": "gym_bro", "insightsEnabled": True,
                    "pushEnabled": True}},
        {"tool": "remember", "result": {"saved": True}},
        {"tool": "forget", "result": {"forgotten": True}},
        # --- nutrition ----------------------------------------------------
        {"tool": "get_macros_today",
         "result": {"date": today,
                    "totals": {"proteinG": 96, "carbsG": 180, "fatG": 52, "calories": 1572},
                    "targets": {"proteinG": 180, "carbsG": 250, "fatG": 70, "calories": 2350},
                    "remaining": {"proteinG": 84, "carbsG": 70, "fatG": 18, "calories": 778},
                    "entries": [{"description": "Greek yogurt and berries", "proteinG": 24},
                                {"description": "Chicken burrito bowl", "proteinG": 52},
                                {"description": "Protein shake", "proteinG": 20}]}},
        {"tool": "lookup_nutrition", "when": {"query": RX(r"(?i).*quest.*")},
         "result": {"results": [{"foodId": FOOD_QUEST,
                                 "description": "Quest Protein Bar, Cookies & Cream",
                                 "servingSize": "1 bar (60 g)", "proteinG": 21,
                                 "carbsG": 22, "fatG": 8, "calories": 190,
                                 "source": "usda"}]}},
        {"tool": "lookup_nutrition", "when": {"query": RX(r"(?i).*egg.*")},
         "result": {"results": [{"foodId": FOOD_EGG, "description": "Egg, whole, large",
                                 "servingSize": "1 large egg (50 g)", "proteinG": 6.3,
                                 "carbsG": 0.4, "fatG": 4.8, "calories": 72,
                                 "source": "usda"}]}},
        {"tool": "lookup_nutrition", "result": {"results": []}},
        {"tool": "lookup_barcode", "when": {"barcode": BARCODE},
         "result": {"found": True, "foodId": FOOD_SHAKE,
                    "name": "Chocolate Protein Shake", "brand": "Shake Co.",
                    "servingSize": "1 bottle (340 mL)", "proteinG": 30, "carbsG": 9,
                    "fatG": 3, "calories": 180, "source": "openfoodfacts"}},
        {"tool": "lookup_barcode", "result": {"found": False}},
        {"tool": "list_saved_meals",
         "result": {"meals": [
             {"id": SM_BREAKFAST, "name": "Usual breakfast", "mealType": "breakfast",
              "items": [{"description": "3 eggs scrambled", "proteinG": 19, "calories": 216},
                        {"description": "2 slices sourdough toast", "proteinG": 8,
                         "calories": 240}]},
             {"id": SM_SHAKE, "name": "Post-workout shake", "mealType": "snack",
              "items": [{"description": "Whey + banana", "proteinG": 26, "calories": 230}]}]}},
        {"tool": "get_meal_plans",
         "result": {"plans": [{"planId": MEAL_PLAN_ID, "name": "Lean bulk", "active": True,
                               "meals": [{"name": "Breakfast"}, {"name": "Lunch"},
                                         {"name": "Dinner"}, {"name": "Snack"}]}]}},
        {"tool": "get_nutrition_profile",
         "result": {"heightCm": 180, "activityLevel": "moderate",
                    "goal": "lean bulk", "calorieTarget": 2350}},
        {"tool": "get_adaptive_targets",
         "result": {"status": "insufficient_data", "daysLogged": 9}},
    ]


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

#: Reads that never change anything — allowed before any answer.
_NOT_FREE_READS = {"remember", "forget"}


class _Builder:
    def __init__(self, snap: dict[str, Any]):
        self.snap = snap
        self.today = snap["today"]
        d = date.fromisoformat(self.today)
        self.yesterday = (d - timedelta(days=1)).isoformat()
        self.samples: list[dict[str, Any]] = []

    def free_reads(self, domain: str) -> list[str]:
        tools = self.snap["by_name"]
        return [n for n in self.snap["domains"][domain]
                if not tools[n]["terminal"] and n not in _NOT_FREE_READS]

    def add(self, num: int, slug: str, category: str, difficulty: str,
            prompt: str, expected: list[dict[str, Any]], notes: str, *,
            domain: str = "general", allow: tuple[str, ...] = (),
            exact: bool = False, ordered: bool = False,
            continue_after: bool = False) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"unknown Spotter category {category!r}")
        names = self.snap["domains"][domain]
        tools = self.snap["by_name"]
        allowed = [n for n in [*self.free_reads(domain), *allow] if n in names]
        self.samples.append({
            "id": f"{num:04d}_{slug}",
            "category": category,
            "difficulty": difficulty,
            "prompt": prompt,
            "system_prompt_refs": {"spotter_router": f"{domain}.spotter_router",
                                   "native": f"{domain}.native"},
            "tool_refs": list(names),
            "expected_tool_calls": expected,
            "scoring": {
                "require_exact_tool_names": True,
                "require_exact_arguments": exact,
                "allow_extra_tool_calls": False,
                "require_order": ordered,
                "allow_argument_coercion": False,
                "trim_string_whitespace": True,
                "allowed_extra_tools": sorted(set(allowed)),
                "validate_arguments_schema": True,
            },
            "max_steps": MAX_STEPS,
            "terminal_tools": [n for n in names if tools[n]["terminal"]],
            "continue_after_terminal": continue_after or domain == "compound",
            "tool_result_set": "fixture_user",
            "notes": f"[{domain} route] {notes}",
        })


def build_spotter_samples(snap: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """All Spotter samples as plain dicts (refs unresolved), sorted by id."""
    snap = snap or load_snapshot()
    b = _Builder(snap)
    today, yday = b.today, b.yesterday
    session = "create_pending_workout_session"

    # ---------------------------------------------------------- logging
    b.add(1, "typos_two_lifts", "workout_logging", "medium",
          "benc 135 fr 8 then sqaut 225 fr 5",
          [call(session, date=OPT(today), exercises=[
              {"name": BENCH, "sets": [S(135, 8)]},
              {"name": SQUAT, "sets": [S(225, 5)]}])],
          "Messy typing from the live eval: misspelled lifts, 'fr' for 'for'. "
          "Two exercises, lb by default, today's date.")
    b.add(2, "yesterday_in_kg", "workout_logging", "medium",
          "yesterday I pulled deadlifts, 3 sets of 5 at 140kg",
          [call(session, date=yday, exercises=[
              {"name": DEADLIFT, "sets": [S(140, 5, "kg")] * 3}])],
          "Relative date → yesterday's ISO date; explicit kg overrides the lb default.")
    b.add(3, "varied_sets_drop", "workout_logging", "medium",
          "squats today: 315 for 5, 315 for 5, then dropped to 275 for 8",
          [call(session, date=OPT(today), exercises=[
              {"name": SQUAT, "sets": [
                  S(315, 5), S(315, 5),
                  S(275, 8, set_type=OPT(ONE_OF("working", "drop", "backoff")))]}])],
          "Three sets with a lighter last set; the drop may be typed drop/backoff.")
    b.add(4, "prompt_example_push_day", "workout_logging", "medium",
          "push day — bench 185 for 5,5,4, incline DB 70s for 8,8,7",
          [call(session, date=OPT(today), exercises=[
              {"name": BENCH, "sets": [S(185, 5), S(185, 5), S(185, 4)]},
              {"name": INCLINE_DB, "sets": [S(70, 8), S(70, 8), S(70, 7)]}])],
          "The exact example from Spotter's own logging instructions.")
    b.add(5, "warmup_set_type", "workout_logging", "medium",
          "bench today: warmup 135x10, then 3 working sets of 185x5",
          [call(session, date=OPT(today), exercises=[
              {"name": BENCH, "sets": [S(135, 10, set_type="warmup"),
                                      S(185, 5), S(185, 5), S(185, 5)]}])],
          "Warm-up set must be typed warmup; working sets expanded to three.")
    b.add(6, "treadmill_distance", "workout_logging", "hard",
          "20 min on the treadmill at 6 mph, 2% incline",
          [call(session, date=OPT(today), exercises=[
              {"name": RX(r"(?i).*treadmill.*"), "sets": [
                  {"durationSeconds": 1200, "speed": 6,
                   "distance": RANGE(1.95, 2.05), "distanceUnit": OPT("mi")}]}])],
          "Cardio rules: minutes → seconds, speed kept, distance = pace × time (2 mi).")
    b.add(7, "stairmaster_intervals", "workout_logging", "hard",
          "stairmaster: 5 min on level 12 then 7 min on level 10",
          [call(session, date=OPT(today), exercises=[
              {"name": RX(r"(?i).*stair.*"), "sets": [
                  {"durationSeconds": 300, "speed": 12, "distance": ABSENT},
                  {"durationSeconds": 420, "speed": 10, "distance": ABSENT}]}])],
          "One set per interval, level → speed, and NO computed distance.")
    b.add(8, "bike_time_only", "workout_logging", "medium",
          "did 20 min on the bike, resistance 12",
          [call(session, date=OPT(today), exercises=[
              {"name": RX(r"(?i).*(bike|cycl).*"), "sets": [
                  {"durationSeconds": 1200, "distance": ABSENT, "speed": ABSENT}]}])],
          "Bike: time is the headline; resistance goes in notes, not speed/distance.")
    b.add(9, "todays_plan_done", "workout_logging", "easy",
          "just got back from the gym, ran through everything on my plan for today",
          [call("log_planned_workout")],
          "Did the planned session without listing sets → log_planned_workout, no "
          "args. Phrased so the planned-workout fast path doesn't catch it.",
          exact=True)

    # ---------------------------------------------------------- history reads
    history_reads = [alt("search_workouts"), alt("get_recent_workouts")]
    b.add(20, "last_bench", "history_reads", "easy",
          "how much did I bench last time?",
          [call("get_exercise_history", alternatives=history_reads)],
          "Must read history (any workout read) and answer — no writes.",
          allow=("show_table",))
    b.add(21, "weight_trend_question", "history_reads", "easy",
          "has my weight been going down?",
          [call("get_bodyweight_history")],
          "Weight questions read get_bodyweight_history; never claim no access.")
    b.add(22, "weekly_summary", "history_reads", "easy",
          "how's my week looking? streak, volume, PRs",
          [call("get_progress_summary")],
          "Progress overview from get_progress_summary.", allow=("show_table",))
    b.add(23, "progression_today", "history_reads", "medium",
          "what weight should I use for rows today?",
          [call("get_progression_suggestions",
                alternatives=[alt("get_exercise_history",
                                  exerciseName=RX(r"(?i).*row.*"))])],
          "'What should I lift' → progression suggestions (history is acceptable).",
          allow=("show_table",))
    b.add(24, "protein_today", "history_reads", "easy",
          "how much protein have I had today?",
          [call("get_macros_today", date=OPT(today),
                alternatives=[alt("show_macro_summary", date=OPT(today))])],
          "Never invent totals: read them.", domain="nutrition")

    # ---------------------------------------------------------- id-grounded edits
    b.add(30, "edit_bench_weight", "id_grounded_edits", "hard",
          "make the bench from yesterday 195 instead of 185",
          [call("create_pending_workout_edit", workoutId=W_YDAY, setUpdates=UNORDERED(
              *({"setId": sid, "weight": 195, "weightUnit": OPT("lb")}
                for sid in (S_B1, S_B2, S_B3))))],
          "Read, then edit all three bench sets by their exact setIds.")
    b.add(31, "fix_last_squat_set", "id_grounded_edits", "hard",
          "the last squat set yesterday was actually 275, not 315",
          [call("create_pending_workout_edit", workoutId=W_YDAY,
                setUpdates=[{"setId": S_Q3, "weight": 275, "weightUnit": OPT("lb")}])],
          "Exactly one setUpdate, on the third squat set.")
    b.add(32, "delete_last_workout", "id_grounded_edits", "medium",
          "delete my last workout",
          [call("create_pending_workout_delete", target="workout", targetId=W_YDAY)],
          "Look up the most recent workout, then delete it by workoutId.")
    b.add(33, "delete_one_set", "id_grounded_edits", "hard",
          "remove the second bench set from yesterday, I didn't actually do it",
          [call("create_pending_workout_delete", target="set", targetId=S_B2)],
          "Delete one set by its setId — not the exercise or the workout.")
    b.add(34, "delete_exercise", "id_grounded_edits", "hard",
          "delete the squats from yesterday's workout",
          [call("create_pending_workout_delete", target="exercise", targetId=EX_SQUAT)],
          "Delete one exercise by workoutExerciseId, keeping the bench.")
    b.add(35, "delete_everything", "id_grounded_edits", "easy",
          "wipe all my workouts, I want to start fresh",
          [call("create_pending_workout_delete", target="all", targetId=ABSENT)],
          "target 'all' with no targetId; no lookup needed.")
    b.add(36, "delete_nonexistent", "id_grounded_edits", "hard",
          "delete my leg press from last week",
          [],
          "No leg press exists in any read result: reads are fine, but deleting "
          "anything is a hallucinated target. Must tell the user instead.")

    # ---------------------------------------------------------- charts
    d = date.fromisoformat(today)

    def ago(days: int) -> str:
        return (d - timedelta(days=days)).isoformat()

    b.add(40, "bench_e1rm_chart", "charts", "hard",
          "chart my bench estimated 1RM over time",
          [call("plot_workout_data", kind="line", series=[{"points": [
              {"x": ago(43), "y": 205}, {"x": ago(29), "y": 210},
              {"x": ago(15), "y": 213}, {"x": ago(1), "y": 216}]}])],
          "Fetch the trend, then plot exactly the fetched points — never invent any.")
    b.add(41, "bodyweight_chart", "charts", "hard",
          "plot my bodyweight over the last month",
          [call("plot_workout_data", kind="line", series=[{"points": [
              {"x": ago(28), "y": 186.2}, {"x": ago(21), "y": 185.4},
              {"x": ago(14), "y": 184.6}, {"x": ago(7), "y": 183.9},
              {"x": today, "y": 183.0}]}])],
          "get_bodyweight_history first, then its {date, weight} points as one series.")

    # ---------------------------------------------------------- nutrition
    macro = "create_pending_macro_log"
    b.add(50, "eggs_breakfast", "nutrition", "easy",
          "log 2 large eggs for breakfast",
          [call(macro, mealType="breakfast",
                items=[{"description": HAS("egg"), "proteinG": RANGE(10, 16),
                        "calories": RANGE(120, 175)}],
                alternatives=[alt(macro, items=[{
                    "description": HAS("egg"), "proteinG": RANGE(10, 16),
                    "calories": RANGE(120, 175), "mealType": "breakfast"}])])],
          "One item with sane estimates (~12 g protein, ~144 kcal), tagged breakfast.",
          domain="nutrition")
    b.add(51, "exact_macros", "nutrition", "easy",
          "add 40g protein, 60g carbs and 20g fat from my shake",
          [call(macro, items=[{"proteinG": 40, "carbsG": 60, "fatG": 20,
                               "isEstimate": False}])],
          "User-stated grams are used verbatim with isEstimate:false.",
          domain="nutrition")
    b.add(52, "multi_meal_one_call", "nutrition", "hard",
          "for breakfast I had oatmeal, lunch was a chicken caesar wrap, and a "
          "protein bar as a snack",
          [call(macro, items=UNORDERED(
              {"description": HAS("oat"), "mealType": "breakfast"},
              {"description": HAS("wrap"), "mealType": "lunch"},
              {"description": HAS("bar"), "mealType": "snack"}))],
          "Several meals → ONE call with per-item mealType; never log just the first.",
          domain="nutrition")
    b.add(53, "branded_lookup", "nutrition", "hard",
          "log a Quest cookies and cream protein bar",
          [call("lookup_nutrition", query=HAS("quest")),
           call(macro, items=[{"foodId": FOOD_QUEST, "proteinG": 21, "calories": 190,
                               "isEstimate": False}])],
          "Branded food: look it up first, then log the grounded numbers with foodId.",
          domain="nutrition", ordered=True)
    b.add(54, "barcode_lookup", "nutrition", "hard",
          f"log this barcode: {BARCODE}",
          [call("lookup_barcode", barcode=BARCODE),
           call(macro, items=[{"foodId": FOOD_SHAKE, "proteinG": 30,
                               "isEstimate": False}])],
          "Barcode → lookup_barcode, then the Open Food Facts numbers.",
          domain="nutrition", ordered=True)
    b.add(55, "protein_goal_only", "nutrition", "easy",
          "can you make my protein goal a bit higher, like 190? keep the rest",
          [call("set_macro_goals", proteinTargetG=190)],
          "Pass ONLY the target mentioned. (The plain 'set my protein goal to "
          "190g' phrasing is answered by the macro-goals fast path, not the model.)",
          domain="nutrition", exact=True)
    b.add(56, "saved_meal", "nutrition", "medium",
          "I had my usual breakfast",
          [call("log_saved_meal", savedMealId=SM_BREAKFAST,
                alternatives=[alt("log_saved_meal", name=CI("usual breakfast"))])],
          "Re-log a saved favorite (by id or exact name), not a new estimate.",
          domain="nutrition")
    b.add(57, "plan_meals_batch", "nutrition", "medium",
          "I just had breakfast and lunch from my meal plan",
          [call("log_plan_meal", mealNames=UNORDERED(CI("breakfast"), CI("lunch")))],
          "Every attested plan meal in ONE log_plan_meal call (mealNames).",
          domain="nutrition")

    # ---------------------------------------------------------- body metrics
    metric = "create_pending_body_metric"
    b.add(60, "weigh_in_today", "body_metrics", "easy",
          "weighed in at 182.4 this morning",
          [call(metric, bodyweight=182.4, bodyweightUnit=OPT("lb"),
                measuredAt=OPT(RX(rf"{today}(T.*)?")),
                alternatives=[alt(metric, entries=[{
                    "bodyweight": 182.4, "bodyweightUnit": OPT("lb"),
                    "measuredAt": OPT(RX(rf"{today}(T.*)?"))}])])],
          "A weigh-in is a body metric, never a workout set. Flat or batch form.")
    b.add(61, "backfill_weigh_ins", "body_metrics", "hard",
          "log my weigh-ins: Oct 1 185.4, Oct 8 184.0, Oct 15 183.2",
          [call(metric, entries=[
              {"measuredAt": RX(r"2026-10-01(T.*)?"), "bodyweight": 185.4,
               "bodyweightUnit": OPT("lb")},
              {"measuredAt": RX(r"2026-10-08(T.*)?"), "bodyweight": 184.0,
               "bodyweightUnit": OPT("lb")},
              {"measuredAt": RX(r"2026-10-15(T.*)?"), "bodyweight": 183.2,
               "bodyweightUnit": OPT("lb")}])],
          "Three dated entries in one call; dates resolved to this year.")

    # ---------------------------------------------------------- device actions
    timer = "control_rest_timer"
    b.add(70, "rest_three_minutes", "device_actions", "easy",
          "give me 3 minutes rest",
          [call(timer, action="start", seconds=180)],
          "Minutes → seconds; the timer lives on the device.")
    b.add(71, "default_timer", "device_actions", "easy",
          "start a timer",
          [call(timer, action="start", seconds=ABSENT)],
          "No duration given → start with NO seconds (device default).")
    b.add(72, "cancel_timer", "device_actions", "easy",
          "kill the timer",
          [call(timer, action="cancel", seconds=ABSENT)],
          "Cancel takes no seconds.")
    b.add(73, "form_video", "device_actions", "easy",
          "show me how to do a romanian deadlift",
          [call("find_exercise_videos",
                query=RX(r"(?i).*(romanian deadlift|rdl).*"), general=OPT(False))],
          "Form help → find_exercise_videos with the movement, general false.")
    b.add(74, "off_topic_video", "device_actions", "easy",
          "find me some lofi study music",
          [call("find_exercise_videos", query=HAS("lofi"), general=True)],
          "Non-fitness videos → general:true; never write URLs.")

    # ---------------------------------------------------------- plans
    b.add(80, "ppl_split", "plans", "hard",
          "make me a PPL split, 6 days a week",
          [call("create_pending_plan", planType="weekly", setActive=OPT(False),
                requestedDaysPerWeek=OPT(6), slots=UNORDERED(
                    {"isRest": True},
                    *({"isRest": OPT(False), "exercises": ANY} for _ in range(6))))],
          "Known template in THIS turn: 7 weekly slots, 6 training + 1 rest, "
          "never activated unasked.")
    b.add(81, "edit_saved_plan", "plans", "hard",
          "add face pulls to my pull day",
          [call("create_pending_plan", planId=PLAN_ID, planType="weekly",
                slots=INCLUDES({"label": RX(r"(?i).*pull.*"), "exercises": INCLUDES(
                    ONE_OF(RX(r"(?i)face ?pulls?"), {"name": RX(r"(?i)face ?pulls?")}))}))],
          "Read the plan, then resend it with its planId and face pulls on Pull.")
    b.add(82, "one_off_session", "plans", "medium",
          "give me a quick 30 minute upper body workout",
          [call("show_workout_plan", exercises=ANY)],
          "A single invented session is a show_workout_plan table, not a saved plan.")
    b.add(83, "plan_today", "plans", "easy",
          "what's on my plan today?",
          [call("show_saved_plan",
                day=OPT(ONE_OF(CI("today"), CI("wednesday"), CI("wed"))))],
          "Their own plan → show_saved_plan; never list it in prose.")

    # ---------------------------------------------------------- settings
    b.add(90, "unit_question", "settings", "easy",
          "what weight unit am I using?",
          [],
          "The unit is in the prompt; a profile read is fine, changing it is not.")
    b.add(91, "switch_to_kg", "settings", "easy",
          "switch me to kg",
          [call("update_user_profile", preferredWeightUnit="kg")],
          "ONLY the field that changes.", exact=True)
    b.add(92, "shorter_replies", "settings", "easy",
          "make your replies shorter",
          [call("update_user_profile", responseLength="brief")],
          "Reply length is a profile setting.", exact=True)

    # ---------------------------------------------------------- clarify / decline
    b.add(100, "bare_load", "clarify_or_decline", "medium", "225 for 5", [],
          "No exercise named: ask which lift; guessing one writes the wrong lift.")
    b.add(101, "advice_question", "clarify_or_decline", "easy",
          "should I log my warmup sets?", [],
          "An advice question, not a log request.")
    b.add(102, "vague_log", "clarify_or_decline", "medium", "log my workout", [],
          "No exercises and no 'today': ask what they did, don't log the plan.")
    b.add(103, "greeting", "clarify_or_decline", "easy", "hey what's up", [],
          "Chat only.")
    b.add(104, "general_knowledge", "clarify_or_decline", "easy",
          "what's a good rep range for hypertrophy?", [],
          "General knowledge — no tools needed.")
    b.add(105, "unsafe_calorie_target", "clarify_or_decline", "hard",
          "set my calories to 800 a day so I can cut fast", [],
          "Safety floor: never propose a dangerously low intake.")

    # ---------------------------------------------------------- memory & privacy
    b.add(110, "injury_memory", "memory_and_privacy", "medium",
          "fyi I tweaked my left knee last week, deep squats hurt",
          [call("remember", content=HAS("knee"), kind="health",
                alternatives=[alt("update_user_profile", injuries=HAS("knee"))])],
          "A durable health fact → remember (health) or the profile's injuries.",
          allow=("remember", "update_user_profile"))
    b.add(111, "no_medication_memory", "memory_and_privacy", "hard",
          "I'm on 20mg of lexapro btw", [],
          "Medications must NEVER be stored — no remember, no profile write.")

    # ---------------------------------------------------------- compound
    b.add(120, "split_and_meal_plan", "compound", "hard",
          "make me a PPL split and a meal plan for a lean bulk",
          [call("create_pending_plan", planType="weekly", setActive=OPT(False),
                slots=[ANY] * 7)],
          "Split THIS turn (7 weekly slots); the meal plan starts with a chip "
          "question, never an invented create_pending_meal_plan.",
          domain="compound")
    b.add(121, "meal_and_lift", "compound", "hard",
          "had 3 eggs for breakfast, then hit squats 275 for 5 sets of 3",
          [call(macro, items=[{"description": HAS("egg"), "proteinG": RANGE(15, 23)}]),
           call(session, date=OPT(today), exercises=[
               {"name": SQUAT, "sets": [S(275, 3)] * 5}])],
          "Meal + lift in one message → a macro card AND a workout card.",
          domain="compound")

    _add_conversations(b, today)
    b.samples.sort(key=lambda s: s["id"])
    return b.samples


def _logged(number: int, weight: float, reps: int, unit: str = "lb") -> dict[str, Any]:
    return {"setNumber": number, "weight": weight, "weightUnit": unit, "reps": reps}


def _repeat(weight: float, reps: int, count: int = 3, unit: str = "lb") -> list[dict[str, Any]]:
    return [_logged(i, weight, reps, unit) for i in range(1, count + 1)]


def _created(date: str, name: Any, sets: list[dict[str, Any]]) -> dict[str, Any]:
    return {"date": date, "name": name, "sets": sets}


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "unchanged_workouts": [W_YDAY, W_PULL, W_PUSH],
        "created_exercises": [],
    }
    state.update(overrides)
    return state


def _add_conversations(b: _Builder, today: str) -> None:
    """Fifteen dialogues. The final world is the gold; the trace is not.

    Every turn is written to stay on the general route and off the no-model
    fast paths. A terminal proposal commits at the end of its turn, so the
    next message can read it back.
    """
    row = RX(r"(?i).*(barbell )?rows?")

    def add(num: int, slug: str, turns: list[str], state: dict[str, Any],
            notes: str) -> None:
        b.add(num, slug, "conversation", "hard", turns[0], [], notes)
        b.samples[-1]["turns"] = turns
        b.samples[-1]["expected_state"] = state

    add(200, "log_then_raise_all", [
        "Log bench press for today: three sets of 5 at 185 pounds.",
        "Make all three of those sets 195 pounds.",
        "Leave yesterday's bench at 185. This is only about the session I just logged.",
    ], _state(created_exercises=[_created(today, BENCH, _repeat(195, 5))]),
        "Log a new bench session, then raise every set. Yesterday stays put. "
        "Any read that finds the new ids is fine.")

    add(201, "only_the_last_set", [
        "Log bench press for today: 185 pounds for 5, then 185 for 5, then 185 for 5.",
        "Change all three of those sets to 195.",
        "Correction: only the last set stays at 195. Put the first two sets back to 185.",
        "Yesterday's workout stays exactly as it was.",
    ], _state(created_exercises=[_created(today, BENCH, [
        _logged(1, 185, 5), _logged(2, 185, 5), _logged(3, 195, 5)])]),
        "A later turn narrows an edit the previous turn applied to every set.")

    add(202, "log_then_delete_it", [
        "Log bench press for today, three sets of 5 at 185 pounds.",
        "Delete that workout I just logged. I added it by mistake.",
        "Yesterday's lower body session stays.",
    ], _state(),
        "The created workout is gone by the end. Fixture sessions stay.")

    yday_bench = {"workoutId": W_YDAY, "exercises": [
        {"workoutExerciseId": EX_BENCH, "sets": [
            {"setId": S_B1, "weight": 185, "reps": 5},
            {"setId": S_B2, "weight": 185, "reps": 5},
            {"setId": S_B3, "weight": 195, "reps": 5}]},
        {"workoutExerciseId": EX_SQUAT, "sets": [
            {"setId": S_Q1, "weight": 315, "reps": 5},
            {"setId": S_Q2, "weight": 315, "reps": 5},
            {"setId": S_Q3, "weight": 315, "reps": 5}]}]}
    add(203, "yesterday_bench_then_narrow", [
        "The bench press from yesterday should be 195 pounds, not 185.",
        "Sorry, only the last bench set was 195. The first two sets were 185.",
        "The squats in that session stay at 315.",
        "Do not log a new workout.",
    ], _state(unchanged_workouts=[W_PULL, W_PUSH], workouts=[yday_bench]),
        "Correct yesterday in place, then undo the first two sets. "
        "Logging a fresh session fails.")

    add(204, "fix_last_squat_across_turns", [
        "The last squat set from yesterday was 275 pounds, not 315.",
        "The first two squat sets really were 315.",
        "Bench press from that day stays at 185.",
    ], _state(unchanged_workouts=[W_PULL, W_PUSH], workouts=[{
        "workoutId": W_YDAY, "exercises": [
            {"workoutExerciseId": EX_BENCH, "sets": [
                {"setId": S_B1, "weight": 185, "reps": 5},
                {"setId": S_B2, "weight": 185, "reps": 5},
                {"setId": S_B3, "weight": 185, "reps": 5}]},
            {"workoutExerciseId": EX_SQUAT, "sets": [
                {"setId": S_Q1, "weight": 315, "reps": 5},
                {"setId": S_Q2, "weight": 315, "reps": 5},
                {"setId": S_Q3, "weight": 275, "reps": 5}]}]}]),
        "One set id changes. The follow-ups must not spread the edit.")

    add(205, "delete_one_set_then_keep_rest", [
        "Remove the second bench press set from yesterday. I didn't do it.",
        "The first and third bench sets stay at 185 for 5.",
        "The squats in that workout stay too.",
    ], _state(unchanged_workouts=[W_PULL, W_PUSH], deleted_ids=[S_B2],
              workouts=[{"workoutId": W_YDAY, "exercises": [
                  {"workoutExerciseId": EX_BENCH, "sets": [
                      {"setId": S_B1, "weight": 185, "reps": 5},
                      {"setId": S_B3, "weight": 185, "reps": 5}]},
                  {"workoutExerciseId": EX_SQUAT, "sets": [
                      {"setId": S_Q1, "weight": 315, "reps": 5},
                      {"setId": S_Q2, "weight": 315, "reps": 5},
                      {"setId": S_Q3, "weight": 315, "reps": 5}]}]}]),
        "Delete one set id. The other sets in that workout stay.")

    add(206, "delete_squats_keep_bench", [
        "Delete the squats from yesterday's workout.",
        "Keep the bench press in that same workout.",
        "Do not delete the whole session.",
    ], _state(unchanged_workouts=[W_PULL, W_PUSH],
              deleted_ids=[EX_SQUAT, S_Q1, S_Q2, S_Q3],
              present_ids=[W_YDAY, EX_BENCH, S_B1, S_B2, S_B3],
              workouts=[{"workoutId": W_YDAY, "exercises": [
                  {"workoutExerciseId": EX_BENCH, "sets": [
                      {"setId": S_B1, "weight": 185, "reps": 5},
                      {"setId": S_B2, "weight": 185, "reps": 5},
                      {"setId": S_B3, "weight": 185, "reps": 5}]}]}]),
        "Drop one exercise. The workout and its bench sets remain.")

    add(207, "overlap_then_edit", [
        "Log bench press for today, three sets of 5 at 185 pounds.",
        "Log bench press for today again, three sets of 5 at 195 pounds.",
        "If the second log was rejected because today already has bench, "
        "edit the existing sets to 195.",
        "There should be one bench session today, not two.",
    ], _state(created_exercises=[_created(today, BENCH, _repeat(195, 5))]),
        "A same-day duplicate is rejected with the existing set ids. "
        "The edit lands on that workout.")

    add(208, "two_lifts_fix_one", [
        "Today I benched 135 pounds for 8 and squatted 225 pounds for 5.",
        "The squat was actually a single set at 235 pounds for 5.",
        "Bench stays one set of 135 for 8.",
        "Don't change any workout from before today.",
    ], _state(created_exercises=[
        _created(today, BENCH, [_logged(1, 135, 8)]),
        _created(today, SQUAT, [_logged(1, 235, 5)]),
    ]),
        "One create or two both pass. The squat weight is the corrected one, "
        "and no 225 set remains.")

    add(209, "bench_then_rows", [
        "Log bench press for today, three sets of 5 at 185 pounds.",
        "I also did barbell rows today, three sets of 8 at 135 pounds. Log those too.",
        "Leave every workout from before today unchanged.",
    ], _state(created_exercises=[
        _created(today, BENCH, _repeat(185, 5)),
        _created(today, row, _repeat(135, 8)),
    ]),
        "A second exercise the same day is not an overlap. "
        "Grouping the two creates is not pinned.")

    add(210, "knee_memory_then_log", [
        "Remember that my left knee is cranky and deep squats hurt.",
        "Log squats for today anyway: three sets of 5 at 225 pounds, nothing deep.",
        "Keep that knee note saved.",
        "Don't change yesterday's squat weights.",
    ], _state(memories=[{"content": HAS("knee"), "kind": "health"}],
              created_exercises=[_created(today, SQUAT, _repeat(225, 5))]),
        "The injury is stored, and today's squats are a new session.")

    add(211, "medication_not_stored", [
        "I'm on 20mg of lexapro, for your context.",
        "Do not save that medication anywhere.",
        "Log bench press for today, two sets of 135 pounds for 8.",
        "The only thing stored from this chat is that bench session.",
    ], _state(absent_memories=["lexapro"], profile={"injuries": None},
              created_exercises=[_created(today, BENCH, [
                  _logged(1, 135, 8), _logged(2, 135, 8)])]),
        "A remembered medication fails even if a later turn logs the bench. "
        "Forgetting it before the end passes.")

    add(212, "kg_switch_then_deadlift", [
        "Switch my weight unit to kilograms.",
        "Log deadlift for today, three sets of 5 at 140 kilograms.",
        "Don't rewrite yesterday's weights. Those stay in pounds.",
        "My unit setting stays kilograms.",
    ], _state(profile={"preferredWeightUnit": "kg"},
              created_exercises=[_created(today, DEADLIFT, _repeat(140, 5, unit="kg"))]),
        "The profile unit flips. Fixture sessions keep their pound loads.")

    add(213, "delete_last_then_spare_older", [
        "Delete my last workout.",
        "Yes, that is yesterday's lower and bench session.",
        "The pull day from earlier stays.",
        "The push day stays too.",
    ], _state(unchanged_workouts=[W_PULL, W_PUSH],
              deleted_ids=[W_YDAY, EX_BENCH, EX_SQUAT, S_B1, S_B2, S_B3,
                           S_Q1, S_Q2, S_Q3]),
        "The newest fixture session goes. Older sessions stay, "
        "whether the delete happens on the first turn or the second.")

    add(214, "forget_knee_remember_shoulder", [
        "Remember that I tweaked my left knee and deep squats hurt.",
        "The knee is fine now. Forget that.",
        "Remember that my right shoulder gets cranky on overhead press.",
        "Don't log a workout.",
    ], _state(memories=[{"content": HAS("shoulder"), "kind": "health"}],
              absent_memories=["knee"]),
        "Memory is part of the world: the knee fact is gone and the "
        "shoulder fact is stored. No workout is created.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_spotter_document(snap: dict[str, Any] | None = None) -> dict[str, Any]:
    snap = snap or load_snapshot()
    samples = build_spotter_samples(snap)
    used = sorted({n for s in samples for n in s["tool_refs"]})
    tools = {
        n: {"type": "function", "function": {
            "name": n,
            "description": snap["by_name"][n]["description"],
            "parameters": snap["by_name"][n]["parameters"]}}
        for n in used
    }
    return {
        "benchmark": BENCHMARK_NAME,
        "suite": "spotter",
        "version": __version__,
        "sample_count": len(samples),
        "categories": list(CATEGORIES),
        "spotter": {
            "gym_bro_commit": snap.get("gym_bro_commit"),
            "today": snap["today"],
            "coach_mode": snap.get("coach_mode"),
            "fixture_profile": snap.get("fixture_profile"),
        },
        "prompts": snap["prompt_text"],
        "tools": tools,
        "tool_result_sets": {"fixture_user": _world(snap["today"])},
        "samples": samples,
    }


def generate_spotter_sample_files(base_dir: str | Path = "benchmark_samples",
                                  snapshot_dir: Path = SNAPSHOT_DIR) -> Path:
    """Write the Spotter samples and spotter_benchmark.json; validate first.

    Per-sample files keep their ``tool_refs`` / ``system_prompt_refs`` /
    ``tool_result_set``; they resolve against the tables in the combined file.
    """
    from .samples import load_benchmark_file

    base = Path(base_dir)
    doc = build_spotter_document(load_snapshot(snapshot_dir))
    combined_path = base / COMBINED_FILENAME
    sample_dir = base / SUITE_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)
    for stale in sample_dir.glob("*.json"):
        stale.unlink()
    for sample in doc["samples"]:
        atomic_write_json(sample_dir / f"{sample['id']}.json", sample)
    atomic_write_json(combined_path, doc)
    load_benchmark_file(combined_path)  # resolves refs and validates
    return combined_path


def validate_spotter_samples(snap: dict[str, Any] | None = None):
    """Resolve and validate in memory (used by tests)."""
    import copy

    from .samples import (
        _resolve_prompt_refs, _resolve_tool_refs, _resolve_tool_result_set,
    )

    doc = build_spotter_document(snap)
    raw = copy.deepcopy(doc["samples"])
    for sample in raw:
        _resolve_prompt_refs(sample, doc["prompts"], Path("<spotter>"))
        _resolve_tool_refs(sample, doc["tools"], Path("<spotter>"))
        _resolve_tool_result_set(sample, doc["tool_result_sets"], Path("<spotter>"))
    return validate_samples(raw, allowed_categories=CATEGORIES)


def dump_prompts_for_route_check(path: Path,
                                 snap: dict[str, Any] | None = None) -> None:
    """Write {sample_id: {prompt, domain}} for the sync script's route check."""
    samples = build_spotter_samples(snap)
    routes: dict[str, dict[str, str]] = {}
    for s in samples:
        domain = s["system_prompt_refs"]["native"].split(".")[0]
        turns = s.get("turns") or [s["prompt"]]
        for index, prompt in enumerate(turns):
            key = s["id"] if index == 0 else f"{s['id']}#{index + 1}"
            routes[key] = {"prompt": prompt, "domain": domain}
    path.write_text(json.dumps(routes, indent=2), encoding="utf-8")

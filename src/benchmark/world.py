"""Fixture world for multi-turn Spotter dialogues.

Single-turn samples answer reads from a static script. A dialogue cannot:
the second turn has to see the workout the first turn logged, the edit the
second turn made, and the delete the third turn committed. This module keeps
that world, applies the write tools the dialogues use, and answers the reads
from the result.

A terminal proposal is committed when the turn ends, which is how the rest
of the suite treats the proposal as the action. The next user message can
then look the write up. Confirm-card taps are not a separate turn.

Scoring compares the final world to the sample's ``expected_state``. Several
call traces reach the same state (search versus recent workouts, one create
versus two); the trace is recorded and not scored.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from .schema import (
    ExpectedState, ProviderResult, Sample, ScoringConfig, ScoringResult,
    ToolCall,
)
from .scoring import _bipartite, compare_arguments

_SUBSET = ScoringConfig(require_exact_arguments=False)

_SET_FIELDS = (
    "setId", "setNumber", "weight", "weightUnit", "reps", "setType",
    "durationSeconds", "distance", "distanceUnit", "rpe", "estimated1rm",
)

_PROPOSAL = {
    "status": "proposal_created",
    "note": "The confirmation card is already showing to the user.",
}


@dataclass
class ApplyResult:
    """What a tool call did to the world."""

    payload: Any
    ends_turn: bool


def _norm(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    drop = {"barbell", "dumbbell", "db", "flat", "back", "conventional",
            "the", "a"}
    return " ".join(part for part in cleaned.split() if part not in drop)


def canonical_movement(name: str) -> str:
    """Fold obvious aliases so a second 'bench' log hits the same exercise."""
    n = _norm(name)
    if "incline" in n:
        return "incline"
    if "bench" in n:
        return "bench"
    if "deadlift" in n or n in ("dl", "rdl"):
        return "rdl" if "romanian" in n or n == "rdl" else "deadlift"
    if "romanian" in n:
        return "rdl"
    if "squat" in n:
        return "squat"
    if "row" in n:
        return "row"
    if "pull" in n and "up" in n:
        return "pullup"
    if "overhead" in n or n == "ohp":
        return "ohp"
    return n


def _names_match(query: str, exercise: str) -> bool:
    q, n = _norm(query), _norm(exercise)
    if not q:
        return True
    if q in n or n in q:
        return True
    return canonical_movement(query) == canonical_movement(exercise)


def _as_entries(results: list[Any]) -> list[tuple[str, dict[str, Any], Any]]:
    out = []
    for item in results:
        if isinstance(item, dict):
            out.append((item.get("tool", ""), item.get("when") or {},
                        item.get("result")))
        else:
            out.append((item.tool, item.when, item.result))
    return out


class World:
    """One fixture user's workouts, memories, and profile."""

    def __init__(self, workouts: list[dict[str, Any]], profile: dict[str, Any],
                 today: str):
        self.workouts = workouts
        self.profile = profile
        self.today = today
        self.memories: list[dict[str, str]] = []
        self.deleted_ids: set[str] = set()
        self._minted = 0

    @classmethod
    def from_tool_results(cls, results: list[Any]) -> "World":
        """Build the fixture from the scripted read results the suite ships."""
        entries = _as_entries(results)
        workouts: list[dict[str, Any]] | None = None
        profile: dict[str, Any] | None = None
        today: str | None = None
        for tool, when, result in entries:
            if not isinstance(result, dict):
                continue
            if tool == "get_recent_workouts" and not when and workouts is None:
                workouts = copy.deepcopy(result.get("workouts") or [])
            elif tool == "get_user_profile" and profile is None:
                profile = copy.deepcopy(result)
            elif tool == "get_macros_today" and today is None:
                today = result.get("date")
        if not workouts or profile is None or not today:
            raise ValueError(
                "fixture tool results are missing workouts, profile, or today")
        for workout in workouts:
            workout["origin"] = "fixture"
        return cls(workouts, profile, today)

    def project(self) -> dict[str, Any]:
        """JSON state the scorer and the report both read."""
        return {
            "today": self.today,
            "workouts": [self._public(w, scored=True) for w in self._live()],
            "deleted_ids": sorted(self.deleted_ids),
            "memories": [dict(m) for m in self.memories],
            "profile": copy.deepcopy(self.profile),
        }

    def apply(self, call: ToolCall) -> ApplyResult | None:
        """Apply ``call``. ``None`` means this tool is not simulated."""
        handler = {
            "get_recent_workouts": self._recent,
            "search_workouts": self._search,
            "get_exercise_history": self._history,
            "get_user_profile": self._profile,
            "remember": self._remember,
            "forget": self._forget,
            "create_pending_workout_session": self._create,
            "create_pending_workout_edit": self._edit,
            "create_pending_workout_delete": self._delete,
            "update_user_profile": self._update_profile,
        }.get(call.name)
        if handler is None:
            return None
        return handler(call.arguments)

    # -- reads ----------------------------------------------------------

    def _live(self) -> list[dict[str, Any]]:
        return [w for w in self.workouts if w["workoutId"] not in self.deleted_ids]

    def _public(self, workout: dict[str, Any], *,
                scored: bool = False) -> dict[str, Any]:
        """The workout a read returns. ``scored`` adds ``origin``, which is
        not part of the production payload."""
        out = {
            "workoutId": workout["workoutId"],
            "date": workout["date"],
            "title": workout.get("title"),
            "exercises": [
                {"workoutExerciseId": ex["workoutExerciseId"],
                 "name": ex["name"],
                 "sets": [self._public_set(s) for s in ex["sets"]]}
                for ex in workout["exercises"]
            ],
        }
        if scored:
            out["origin"] = workout.get("origin", "fixture")
        return out

    def _public_set(self, s: dict[str, Any]) -> dict[str, Any]:
        out = {key: s.get(key) for key in _SET_FIELDS if key in s or key in (
            "setId", "setNumber", "weight", "weightUnit", "reps", "setType")}
        return out

    def _recent(self, args: dict[str, Any]) -> ApplyResult:
        limit = args.get("limit")
        if not isinstance(limit, int) or isinstance(limit, bool):
            limit = 3
        ordered = sorted(self._live(), key=lambda w: (w["date"], w["workoutId"]),
                         reverse=True)
        chosen = ordered[:limit]
        return ApplyResult({
            "weightUnit": self.profile.get("preferredWeightUnit", "lb"),
            "count": len(chosen),
            "workouts": [self._public(w) for w in chosen],
        }, ends_turn=False)

    def _search(self, args: dict[str, Any]) -> ApplyResult:
        matched = []
        for workout in self._live():
            if not self._in_range(workout["date"], args.get("from"), args.get("to")):
                continue
            name = args.get("exerciseName") or ""
            query = args.get("query") or ""
            exercises = workout["exercises"]
            if name and not any(_names_match(name, ex["name"]) for ex in exercises):
                continue
            if query and not (
                    _names_match(query, workout.get("title") or "")
                    or any(_names_match(query, ex["name"]) for ex in exercises)):
                continue
            matched.append(workout)
        matched.sort(key=lambda w: (w["date"], w["workoutId"]), reverse=True)
        limit = args.get("limit") if isinstance(args.get("limit"), int) else 25
        returned = matched[:limit]
        return ApplyResult({
            "weightUnit": self.profile.get("preferredWeightUnit", "lb"),
            "totalMatches": len(matched),
            "returned": len(returned),
            "workouts": [self._public(w) for w in returned],
        }, ends_turn=False)

    def _history(self, args: dict[str, Any]) -> ApplyResult:
        query = str(args.get("exerciseName") or "")
        sessions = []
        for workout in sorted(self._live(),
                              key=lambda w: w["date"], reverse=True):
            for ex in workout["exercises"]:
                if not _names_match(query, ex["name"]):
                    continue
                sessions.append({
                    "workoutId": workout["workoutId"],
                    "date": workout["date"],
                    "sets": [{"weight": s.get("weight"), "reps": s.get("reps"),
                              "durationSeconds": s.get("durationSeconds"),
                              "estimated1rm": s.get("estimated1rm")}
                             for s in ex["sets"]],
                })
        if not sessions:
            return ApplyResult({"found": False, "query": query}, ends_turn=False)
        return ApplyResult({
            "found": True,
            "weightUnit": self.profile.get("preferredWeightUnit", "lb"),
            "history": {"exerciseName": query, "recentSessions": sessions},
        }, ends_turn=False)

    def _profile(self, _args: dict[str, Any]) -> ApplyResult:
        return ApplyResult(copy.deepcopy(self.profile), ends_turn=False)

    # -- writes ---------------------------------------------------------

    def _mint(self) -> str:
        self._minted += 1
        return f"c0ffee00-0000-4000-8000-{self._minted:012d}"

    def _unit(self, explicit: Any) -> str:
        if explicit in ("lb", "kg"):
            return explicit
        unit = self.profile.get("preferredWeightUnit")
        return unit if unit in ("lb", "kg") else "lb"

    def _create(self, args: dict[str, Any]) -> ApplyResult:
        date = args.get("date") or self.today
        exercises = args.get("exercises") or []
        overlap = self._overlap(date, exercises)
        if overlap is not None:
            return ApplyResult(overlap, ends_turn=False)
        workout_id = self._mint()
        built = []
        for ex in exercises:
            sets = []
            for i, raw in enumerate(ex.get("sets") or [], start=1):
                number = raw.get("setNumber")
                if not isinstance(number, int) or isinstance(number, bool) or number < 1:
                    number = i
                weight = raw.get("weight")
                sets.append({
                    "setId": self._mint(),
                    "setNumber": number,
                    "weight": weight,
                    "weightUnit": self._unit(raw.get("weightUnit"))
                    if weight is not None else None,
                    "reps": raw.get("reps"),
                    "setType": raw.get("setType") or "working",
                    "durationSeconds": raw.get("durationSeconds"),
                    "distance": raw.get("distance"),
                    "distanceUnit": raw.get("distanceUnit"),
                })
            built.append({"workoutExerciseId": self._mint(),
                          "name": ex.get("name") or "Exercise", "sets": sets})
        self.workouts.append({
            "workoutId": workout_id,
            "date": date,
            "title": args.get("title"),
            "origin": "created",
            "exercises": built,
        })
        return ApplyResult(dict(_PROPOSAL), ends_turn=True)

    def _overlap(self, date: str, exercises: list[Any]) -> dict[str, Any] | None:
        for ex in exercises:
            key = canonical_movement(str(ex.get("name") or ""))
            for workout in self._live():
                if workout["date"] != date:
                    continue
                for existing in workout["exercises"]:
                    if canonical_movement(existing["name"]) != key:
                        continue
                    return {
                        "status": "same_day_overlap",
                        "workoutId": workout["workoutId"],
                        "exerciseName": existing["name"],
                        "sets": [self._public_set(s) for s in existing["sets"]],
                        "note": "This date already has this exercise. Call "
                                "create_pending_workout_edit with these ids "
                                "instead of logging a duplicate.",
                    }
        return None

    def _find_workout(self, workout_id: str) -> dict[str, Any] | None:
        for workout in self._live():
            if workout["workoutId"] == workout_id:
                return workout
        return None

    def _edit(self, args: dict[str, Any]) -> ApplyResult:
        workout = self._find_workout(str(args.get("workoutId") or ""))
        if workout is None:
            return ApplyResult(
                {"error": "not_found",
                 "message": "No workout has that workoutId."},
                ends_turn=False)
        sets = {s["setId"]: s for ex in workout["exercises"] for s in ex["sets"]}
        updates = args.get("setUpdates") or []
        missing = [u.get("setId") for u in updates
                   if u.get("setId") not in sets]
        if missing:
            return ApplyResult(
                {"error": "not_found",
                 "message": "Those setIds are not in this workout.",
                 "setIds": missing},
                ends_turn=False)
        for update in updates:
            target = sets[update["setId"]]
            for key in ("weight", "weightUnit", "reps", "setType",
                        "durationSeconds", "distance", "rpe"):
                if key in update and update[key] is not None:
                    target[key] = update[key]
        session = args.get("sessionUpdates") or {}
        if session.get("title"):
            workout["title"] = session["title"]
        if session.get("workoutDate"):
            workout["date"] = session["workoutDate"]
        return ApplyResult(dict(_PROPOSAL), ends_turn=True)

    def _delete(self, args: dict[str, Any]) -> ApplyResult:
        target = args.get("target")
        target_id = args.get("targetId")
        if target == "all":
            for workout in self._live():
                self._drop_workout(workout)
            return ApplyResult(dict(_PROPOSAL), ends_turn=True)
        if target == "workout":
            workout = self._find_workout(str(target_id or ""))
            if workout is None:
                return ApplyResult(
                    {"error": "not_found",
                     "message": "No workout has that id."}, ends_turn=False)
            self._drop_workout(workout)
            return ApplyResult(dict(_PROPOSAL), ends_turn=True)
        if target == "exercise":
            for workout in self._live():
                for ex in list(workout["exercises"]):
                    if ex["workoutExerciseId"] == target_id:
                        self._drop_exercise(ex)
                        workout["exercises"].remove(ex)
                        return ApplyResult(dict(_PROPOSAL), ends_turn=True)
            return ApplyResult(
                {"error": "not_found",
                 "message": "No exercise has that id."}, ends_turn=False)
        if target == "set":
            for workout in self._live():
                for ex in workout["exercises"]:
                    for s in list(ex["sets"]):
                        if s["setId"] == target_id:
                            self.deleted_ids.add(s["setId"])
                            ex["sets"].remove(s)
                            return ApplyResult(dict(_PROPOSAL), ends_turn=True)
            return ApplyResult(
                {"error": "not_found", "message": "No set has that id."},
                ends_turn=False)
        return ApplyResult(
            {"error": "invalid_arguments",
             "message": "target must be workout, exercise, set, or all."},
            ends_turn=False)

    def _drop_workout(self, workout: dict[str, Any]) -> None:
        self.deleted_ids.add(workout["workoutId"])
        for ex in workout["exercises"]:
            self._drop_exercise(ex)

    def _drop_exercise(self, ex: dict[str, Any]) -> None:
        self.deleted_ids.add(ex["workoutExerciseId"])
        for s in ex["sets"]:
            self.deleted_ids.add(s["setId"])

    def _remember(self, args: dict[str, Any]) -> ApplyResult:
        content = str(args.get("content") or "").strip()
        kind = args.get("kind") or "other"
        folded = content.casefold()
        if content and not any(m["content"].casefold() == folded
                               for m in self.memories):
            self.memories.append({"content": content, "kind": kind})
        return ApplyResult({"saved": True}, ends_turn=False)

    def _forget(self, args: dict[str, Any]) -> ApplyResult:
        content = str(args.get("content") or "").strip()
        folded = content.casefold()
        kept = []
        removed = 0
        for memory in self.memories:
            have = memory["content"].casefold()
            if folded and (folded in have or have in folded):
                removed += 1
            else:
                kept.append(memory)
        self.memories = kept
        return ApplyResult({"forgotten": removed > 0, "removed": removed},
                           ends_turn=False)

    def _update_profile(self, args: dict[str, Any]) -> ApplyResult:
        for key, value in args.items():
            if value is not None:
                self.profile[key] = value
        return ApplyResult(dict(_PROPOSAL), ends_turn=True)

    @staticmethod
    def _in_range(date: str, start: Any, end: Any) -> bool:
        if isinstance(start, str) and date < start:
            return False
        if isinstance(end, str) and date > end:
            return False
        return True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _signature(workout: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": workout["date"],
        "title": workout["title"],
        "exercises": [
            {"id": ex["workoutExerciseId"], "name": ex["name"],
             "sets": [{"id": s.get("setId"), "n": s.get("setNumber"),
                       "weight": s.get("weight"), "unit": s.get("weightUnit"),
                       "reps": s.get("reps"), "type": s.get("setType")}
                      for s in ex["sets"]]}
            for ex in workout["exercises"]
        ],
    }


def _created_pool(state: dict[str, Any]) -> list[dict[str, Any]]:
    pool = []
    for workout in state["workouts"]:
        if workout.get("origin") != "created":
            continue
        for ex in workout["exercises"]:
            pool.append({
                "date": workout["date"],
                "name": ex["name"],
                "sets": [{"setNumber": s.get("setNumber"),
                          "weight": s.get("weight"),
                          "weightUnit": s.get("weightUnit"),
                          "reps": s.get("reps"),
                          "setType": s.get("setType")}
                         for s in ex["sets"]],
            })
    return pool


def _live_ids(state: dict[str, Any]) -> set[str]:
    found = set()
    for workout in state["workouts"]:
        found.add(workout["workoutId"])
        for ex in workout["exercises"]:
            found.add(ex["workoutExerciseId"])
            for s in ex["sets"]:
                if s.get("setId"):
                    found.add(s["setId"])
    return found


def _first_change(before: dict[str, Any], after: dict[str, Any]) -> str:
    """One short phrase naming how a fixture workout moved."""
    b_sets = {s["id"]: s for ex in before["exercises"] for s in ex["sets"]}
    a_sets = {s["id"]: s for ex in after["exercises"] for s in ex["sets"]}
    for sid, old in b_sets.items():
        new = a_sets.get(sid)
        if new is None:
            return f"set {sid} removed"
        if old["weight"] != new["weight"]:
            return f"set {sid} weight {old['weight']} → {new['weight']}"
        if old != new:
            return f"set {sid} changed"
    extra = sorted(set(a_sets) - set(b_sets))
    if extra:
        return f"set {extra[0]} added"
    if before["date"] != after["date"]:
        return f"date {before['date']} → {after['date']}"
    if before["title"] != after["title"]:
        return "title changed"
    return "exercises changed"


def _join(errors: Any, limit: int = 6) -> str:
    items = [str(e) for e in errors]
    shown = items[:limit]
    if len(items) > limit:
        shown.append(f"+{len(items) - limit} more")
    return ", ".join(shown)


def diff_world(expected: ExpectedState, final: dict[str, Any],
               initial: dict[str, Any]) -> list[str]:
    """Human-readable mismatches. Empty means the world matches."""
    errors: list[str] = []
    by_final = {w["workoutId"]: w for w in final["workouts"]}
    by_initial = {w["workoutId"]: w for w in initial["workouts"]}
    deleted = set(final["deleted_ids"])
    live = _live_ids(final)

    for wid in expected.unchanged_workouts:
        if wid not in by_initial:
            errors.append(f"unchanged workout {wid} is not in the fixture")
            continue
        if wid not in by_final:
            errors.append(f"fixture workout {wid} was deleted")
            continue
        before, after = _signature(by_initial[wid]), _signature(by_final[wid])
        if before != after:
            errors.append(
                f"fixture workout {wid} changed ({_first_change(before, after)})")

    for spec in expected.workouts:
        wid = spec.get("workoutId")
        if not isinstance(wid, str):
            errors.append("workout expectation is missing workoutId")
            continue
        actual = by_final.get(wid)
        if actual is None:
            errors.append(f"workout {wid} is missing")
            continue
        found = compare_arguments("workout", spec, actual, _SUBSET)
        if found:
            errors.append(f"workout {wid}: {_join(f'{e.argument_path} ({e.error_type})' for e in found)}")

    for ident in expected.deleted_ids:
        if ident not in deleted:
            errors.append(f"{ident} was not deleted")
    for ident in expected.present_ids:
        if ident not in live:
            errors.append(f"{ident} is not in the final world")

    if expected.created_exercises is not None:
        errors.extend(_match_created(expected.created_exercises,
                                     _created_pool(final)))

    used: set[int] = set()
    for spec in expected.memories:
        hit = next((i for i, mem in enumerate(final["memories"])
                    if i not in used
                    and not compare_arguments("memory", spec, mem, _SUBSET)),
                   None)
        if hit is None:
            errors.append("memory missing: "
                          + json.dumps(spec, ensure_ascii=False)[:160])
        else:
            used.add(hit)

    injuries = final["profile"].get("injuries")
    injury_text = injuries if isinstance(injuries, str) else ""
    for fragment in expected.absent_memories:
        folded = fragment.casefold()
        for memory in final["memories"]:
            if folded in memory["content"].casefold():
                errors.append(
                    f"memory should not contain {fragment!r}: "
                    f"{memory['content']!r}")
        if folded in injury_text.casefold():
            errors.append(f"profile.injuries should not contain {fragment!r}")

    if expected.profile:
        found = compare_arguments("profile", expected.profile,
                                  final["profile"], _SUBSET)
        if found:
            errors.append("profile: " + _join(
                f"{e.argument_path} ({e.error_type})" for e in found))
    return errors


def _match_created(expected: list[dict[str, Any]],
                   pool: list[dict[str, Any]]) -> list[str]:
    pairs = _bipartite(
        len(expected), len(pool),
        lambda i, j: not compare_arguments(
            "created", expected[i], pool[j], _SUBSET))
    errors = []
    matched = set(pairs.values())
    for i in range(len(expected)):
        if i not in pairs:
            errors.append("created exercise missing: "
                          + json.dumps(expected[i], ensure_ascii=False)[:180])
    for j, item in enumerate(pool):
        if j not in matched:
            errors.append(
                f"unexpected created exercise {item['name']!r} on {item['date']} "
                f"({len(item['sets'])} sets)")
    return errors


def score_world_result(sample: Sample, result: ProviderResult) -> ScoringResult:
    """Pass iff the dialogue's final world matches ``expected_state``.

    Timeout, provider error, and protocol error still fail the run. The
    calls themselves are not compared to a gold trace.
    """
    if result.timed_out:
        return ScoringResult(success=False, reason="Timed out.")
    if result.error is not None:
        return ScoringResult(success=False,
                             reason=f"Provider error: {result.error}")
    if result.protocol_error is not None:
        return ScoringResult(
            success=False, protocol_error=result.protocol_error,
            reason=f"Protocol error: {result.protocol_error}")
    assert sample.expected_state is not None
    mismatches = diff_world(sample.expected_state, result.world_state or {},
                            result.initial_world or {})
    if not mismatches:
        return ScoringResult(success=True, reason="Passed. Final world matches.")
    shown = mismatches[:8]
    extra = f"; +{len(mismatches) - 8} more" if len(mismatches) > 8 else ""
    return ScoringResult(
        success=False, world_errors=mismatches,
        reason="Failed — world mismatch: " + "; ".join(shown) + extra + ".",
    )

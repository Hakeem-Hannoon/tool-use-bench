"""Spotter's JSON-router tool protocol, ported from gym-bro.

Spotter does not send a ``tools`` parameter. The tool catalog is plain text
in the system prompt, the request runs in JSON mode, and every step the model
returns ONE object — ``{"kind": "tool", "tool": ..., "arguments": {...}}`` to
run a tool or ``{"kind": "reply", "message": ...}`` to answer. Read-tool
results come back as a ``TOOL_RESULT <name>:`` user message.

The parsing rules below mirror ``apps/web/lib/agent/orchestrator.ts``
(``parseRouterOutput``, ``looksLikeRouterJson``, ``canonicalizeToolName``,
``closeToolNames``) and the nudge wording of ``runRouterLoop`` so a model's
score here predicts its behaviour in production. Spotter's heuristic reply
guards (claim, compound, bare-load, delete, table checks) are deliberately
not replayed: the benchmark measures the model's own decisions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

#: Router nudge after a response that tried to be JSON but didn't parse.
INVALID_JSON_NUDGE = (
    "Your last message was not valid JSON — it may have been cut off. Reply "
    "again with the COMPLETE JSON object only, and keep it compact."
)

#: Follow-up after a non-terminal tool result.
CONTINUE_SUFFIX = "Continue. Respond with the JSON object only."

#: Follow-up after a terminal (proposal) tool when the episode continues.
PROPOSAL_CREATED_RESULT = (
    '{"status":"proposal_created","note":"The confirmation card is already '
    'showing to the user."}'
)
PROPOSAL_CONTINUE_SUFFIX = (
    "If my original request asked for ANOTHER deliverable (e.g. a meal plan "
    "AND a workout split), call that tool now. Otherwise reply with one short "
    "wrap-up sentence (do not repeat the proposals, do not create anything I "
    "didn't ask for). Respond with the JSON object only."
)

#: Used only when a sample doesn't carry its own router system prompt
#: (e.g. running the generic suite under the router protocol). Copied from
#: ROUTER_OUTPUT_FORMAT in apps/web/lib/agent/prompts.ts.
ROUTER_LOOP = (
    "You work in a loop. Each step you either run ONE tool or reply. When you "
    'run a read tool, its result comes back to you as a "TOOL_RESULT" message '
    "— use it to reason, then either reply or run the next tool. Don't guess "
    "at data you can fetch."
)
ROUTER_OUTPUT_FORMAT = """Output format — you MUST respond with a single JSON object and nothing else, in exactly this shape:
{
  "kind": "reply" | "tool",
  "responseType": "chat_text" | "clarification_request",   // only when kind="reply"
  "message": "text to show the user",                        // only when kind="reply"
  "options": ["optional", "quick", "replies"],               // for clarification_request: 2–6 short tap labels
  "tool": "<tool name>",                                     // only when kind="tool"
  "arguments": { }                                           // only when kind="tool"
}

Reply (kind="reply") when you can answer directly or need clarification. Use a tool (kind="tool") when the request needs one. Never include any text outside the JSON object."""

_FENCED = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_LEADING_FENCE = re.compile(r"^```(?:json)?", re.IGNORECASE)
_ROUTER_KEY = re.compile(r'"(?:kind|tool|arguments)"\s*:')
_SVG = re.compile(r"^<svg[\s>]", re.IGNORECASE)


@dataclass(frozen=True)
class RouterEnvelope:
    kind: str  # "reply" | "tool"
    tool: str | None = None
    arguments: Any = None
    response_type: str | None = None
    message: str | None = None


def _valid_envelope(obj: Any) -> RouterEnvelope | None:
    """Validate like Spotter's routerOutputSchema (unknown keys are ignored)."""
    if not isinstance(obj, dict):
        return None
    kind = obj.get("kind")
    if kind not in ("reply", "tool"):
        return None
    rtype = obj.get("responseType")
    if rtype is not None and rtype not in ("chat_text", "clarification_request"):
        return None
    message = obj.get("message")
    if message is not None and not isinstance(message, str):
        return None
    options = obj.get("options")
    if options is not None and (not isinstance(options, list) or len(options) > 6
                                or not all(isinstance(o, str) for o in options)):
        return None
    placeholder = obj.get("freeTextPlaceholder")
    if placeholder is not None and (not isinstance(placeholder, str)
                                    or len(placeholder) > 80):
        return None
    tool = obj.get("tool")
    if tool is not None and not isinstance(tool, str):
        return None
    return RouterEnvelope(kind=kind, tool=tool, arguments=obj.get("arguments"),
                          response_type=rtype, message=message)


def parse_router_output(text: str) -> RouterEnvelope | None:
    """Tolerant envelope extraction — strips code fences and surrounding prose."""
    if not text:
        return None
    fenced = _FENCED.search(text)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(candidate[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return _valid_envelope(obj)


def looks_like_router_json(text: str) -> bool:
    """Did the model ATTEMPT the envelope (vs. answer in plain prose)?"""
    trimmed = _LEADING_FENCE.sub("", text.strip()).lstrip()
    return (trimmed.startswith("{") or trimmed.startswith("[")
            or _ROUTER_KEY.search(text) is not None
            or _SVG.match(trimmed) is not None)


def canonicalize_tool_name(name: str, names: list[str] | set[str]) -> str | None:
    """Map a near-miss to a catalog name: exact, case-insensitive, or a unique
    plural/singular match (``search_workout`` → ``search_workouts``)."""
    raw = name.strip()
    if not raw:
        return None
    ordered = sorted(names) if isinstance(names, set) else list(names)
    if raw in ordered:
        return raw
    lower = raw.lower()
    same = next((n for n in ordered if n.lower() == lower), None)
    if same:
        return same
    plural = [n for n in ordered
              if n.lower() in (f"{lower}s", f"{lower}es")
              or lower in (f"{n.lower()}s", f"{n.lower()}es")]
    return plural[0] if len(plural) == 1 else None


def close_tool_names(name: str, names: list[str] | set[str]) -> list[str]:
    """Catalog names that contain the typo (or vice versa), for the retry hint."""
    lower = re.sub(r"[^a-z0-9_]", "", name.lower())
    if len(lower) < 5:
        return []
    ordered = sorted(names) if isinstance(names, set) else list(names)
    return [n for n in ordered if lower in n.lower() or n.lower() in lower][:5]


def unknown_tool_nudge(tool: str, names: list[str] | set[str]) -> str:
    close = close_tool_names(tool, names) if tool else []
    return (
        f'UNKNOWN TOOL: "{tool}" is not a tool. Nothing was changed. '
        + (f"Closest names: {', '.join(close)}. " if close else "")
        + "Call a tool from the system prompt, spelled exactly. Respond with "
          "the JSON object only."
    )


def tool_result_message(tool: str, payload: str) -> str:
    return f"TOOL_RESULT {tool}:\n{payload}\n\n{CONTINUE_SUFFIX}"


def proposal_result_message(tool: str) -> str:
    return (f"TOOL_RESULT {tool}:\n{PROPOSAL_CREATED_RESULT}\n\n"
            f"{PROPOSAL_CONTINUE_SUFFIX}")


def build_router_system_prompt(base: str, tools: list[dict[str, Any]],
                               *, terminal: set[str] | None = None) -> str:
    """Generic router prompt for samples without a Spotter-exported one.

    Spotter's catalog lines carry argument hints in prose; generic tools only
    have JSON Schema, so each line includes the compact parameter schema.
    """
    terminal = terminal or set()
    lines = []
    for tool in tools:
        fn = tool.get("function", {})
        params = json.dumps(fn.get("parameters", {}), separators=(",", ":"),
                            ensure_ascii=False)
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')} "
                     f"Arguments (JSON Schema): {params}")
    parts = [p for p in (base.strip(), ROUTER_LOOP,
                         "Available tools:\n" + "\n".join(lines),
                         ROUTER_OUTPUT_FORMAT) if p]
    return "\n\n".join(parts)

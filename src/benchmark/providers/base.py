"""Provider abstraction.

Every adapter exposes one model call as a conversation *step*: send the
history, get back native tool calls, text, and token usage. The episode
runner (:mod:`benchmark.episode`) drives steps — one for a classic
single-turn sample, several when a sample feeds scripted tool results back —
and normalizes everything into :class:`benchmark.schema.ProviderResult`.

Only provider-native tool/function calls count as tool calls under the
``native`` protocol; JSON printed as text is ignored unless
``allow_text_tool_call_fallback`` is enabled. The ``spotter_router``
protocol reads tool calls from a JSON envelope instead (see
:mod:`benchmark.router`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelConfig
from ..schema import ProviderResult, Sample, ToolCall, Usage
from ..utils.env import get_env, require_env
from ..utils.json_utils import try_parse_json


@dataclass(frozen=True)
class ProviderSettings:
    """Run-wide knobs shared by all adapters."""

    temperature: float | None = 0
    allow_text_tool_call_fallback: bool = False
    max_output_tokens: int = 4096


class ProviderCallError(RuntimeError):
    """A provider call failed. ``retryable`` guides the runner's retry loop."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_RETRYABLE_NAME_HINTS = (
    "timeout", "connection", "overloaded", "ratelimit", "rate_limit",
    "serviceunavailable", "internalserver", "apiconnection",
)


def classify_exception(exc: BaseException) -> ProviderCallError:
    """Map an arbitrary SDK exception onto a ProviderCallError with a
    retryability verdict, without depending on any one SDK's class tree."""
    status = None
    for attr in ("status_code", "http_status", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            status = value
            break
    message = f"{type(exc).__name__}: {exc}"
    if status is not None:
        retryable = status in _RETRYABLE_STATUS
        return ProviderCallError(message, retryable=retryable)
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    retryable = any(h in name or h in text for h in _RETRYABLE_NAME_HINTS)
    return ProviderCallError(message, retryable=retryable)


def normalize_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a provider's argument payload into a dict.

    Providers ship arguments either as a JSON string (OpenAI-style) or a
    mapping (Anthropic ``input``, Gemini ``args``). Unparseable payloads are
    preserved under a sentinel key so the run fails scoring deterministically
    instead of crashing the benchmark.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if raw.strip() == "":
            return {}
        parsed = try_parse_json(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"_unparseable_arguments": raw}
    return {"_unparseable_arguments": repr(raw)}


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _as_text_tool_call(obj: Any) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        args = try_parse_json(args) or {}
    if not isinstance(args, dict):
        return None
    return ToolCall(name=name, arguments=args)


def parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Best-effort extraction of JSON tool calls embedded in plain text.

    Used ONLY when ``allow_text_tool_call_fallback`` is true. The default
    benchmark behavior treats text-only responses as containing zero tool
    calls, which fails any sample that expects one.
    """
    if not text:
        return []
    candidates: list[Any] = []
    for block in _FENCE_RE.findall(text):
        parsed = try_parse_json(block.strip())
        if parsed is not None:
            candidates.append(parsed)
    whole = try_parse_json(text.strip())
    if whole is not None:
        candidates.append(whole)
    calls: list[ToolCall] = []
    for candidate in candidates:
        items = candidate if isinstance(candidate, list) else [candidate]
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("tool_calls"), list):
                items_inner = item["tool_calls"]
            else:
                items_inner = [item]
            for inner in items_inner:
                call = _as_text_tool_call(inner)
                if call is not None:
                    calls.append(call)
        if calls:
            break  # first parseable candidate wins; keep deterministic
    return calls


@dataclass
class NativeCall:
    """A tool call as the provider emitted it, with the id needed to answer it."""

    id: str | None
    name: str
    arguments: dict[str, Any]
    #: The provider's original argument payload (e.g. OpenAI's JSON string),
    #: echoed back verbatim when the call is replayed in history.
    raw_arguments: Any = None


@dataclass
class StepOutput:
    """One model response."""

    raw: Any
    calls: list[NativeCall]
    text: str
    usage: Usage = field(default_factory=Usage)
    #: Provider-private assistant message to append to history (keeps fields
    #: such as Gemini thought signatures or Anthropic thinking blocks intact).
    native: Any = None


@dataclass
class Conversation:
    """Provider-native message history for one episode."""

    system: str
    messages: list[Any] = field(default_factory=list)


class BaseProvider:
    """Common adapter interface. Subclasses implement :meth:`step` and the
    three message builders for their wire format."""

    name: str = "base"

    def __init__(self, cfg: ModelConfig, settings: ProviderSettings):
        self.cfg = cfg
        self.settings = settings

    # -- credential / construction hooks --------------------------------

    def check_credentials(self) -> None:
        """Raise MissingCredentialError before any API call if credentials
        are absent. Default: require ``api_key_env`` (and ``endpoint_env``
        when the endpoint comes from the environment)."""
        if self.cfg.api_key_env:
            require_env(self.cfg.api_key_env)
        if self.cfg.endpoint_env and not self.cfg.endpoint:
            require_env(self.cfg.endpoint_env)

    def endpoint(self) -> str | None:
        """The configured endpoint, or the value of ``endpoint_env``."""
        if self.cfg.endpoint:
            return self.cfg.endpoint
        if self.cfg.endpoint_env:
            return get_env(self.cfg.endpoint_env)
        return None

    # -- request construction helpers ------------------------------------

    def request_params(self, **base: Any) -> dict[str, Any]:
        """Merge computed kwargs with per-model ``api_params`` overrides.

        ``api_params`` wins; a JSON ``null`` value removes the key entirely
        (useful for models that reject ``temperature``).
        """
        params = dict(base)
        for key, value in self.cfg.api_params.items():
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
        return params

    # -- conversation primitives (implemented per provider) --------------

    def new_conversation(self, system: str, prompt: str) -> Conversation:
        return Conversation(system=system, messages=[self.user_message(prompt)])

    def user_message(self, text: str) -> Any:
        raise NotImplementedError

    def assistant_message(self, step: StepOutput) -> Any:
        raise NotImplementedError

    def tool_result_messages(self, step: StepOutput,
                             results: list[tuple[NativeCall, str]]) -> list[Any]:
        """Messages answering ``step``'s native calls with JSON payloads."""
        raise NotImplementedError

    async def step(self, conv: Conversation, tools: list[dict[str, Any]] | None,
                   *, json_mode: bool = False) -> StepOutput:
        """Perform one provider API call on ``conv``.

        ``tools`` are OpenAI-style definitions (``None`` = send no tools, as
        the router protocol does). ``json_mode`` requests a JSON-object
        response where the provider supports it. Must raise (anything) on
        failure; the runner classifies and retries.
        """
        raise NotImplementedError

    # -- the episode ------------------------------------------------------

    async def complete(self, sample: Sample, run_index: int,
                       system_prompt: str) -> ProviderResult:
        """Run one episode of ``sample`` and return the normalized result."""
        from ..episode import run_episode  # avoid an import cycle

        return await run_episode(self, sample, run_index, system_prompt)


def safe_dump(response: Any) -> Any:
    """Convert an SDK response object to plain JSON-serializable data."""
    for method in ("model_dump", "to_dict"):
        fn = getattr(response, method, None)
        if callable(fn):
            try:
                return fn(mode="json") if method == "model_dump" else fn()
            except TypeError:
                try:
                    return fn()
                except Exception:  # pragma: no cover - defensive
                    pass
            except Exception:  # pragma: no cover - defensive
                pass
    if isinstance(response, (dict, list, str, int, float, bool)) or response is None:
        return response
    return repr(response)

"""Provider abstraction.

Every adapter turns a benchmark sample into one provider API call and
normalizes the response into :class:`benchmark.schema.ProviderResult`.
Only provider-native tool/function calls count as tool calls; JSON printed
as text is ignored unless ``allow_text_tool_call_fallback`` is enabled.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig
from ..schema import ProviderResult, Sample, ToolCall
from ..utils.env import require_env
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


class BaseProvider(ABC):
    """Common adapter interface. Subclasses implement one API call."""

    name: str = "base"

    def __init__(self, cfg: ModelConfig, settings: ProviderSettings):
        self.cfg = cfg
        self.settings = settings

    # -- credential / construction hooks --------------------------------

    def check_credentials(self) -> None:
        """Raise MissingCredentialError before any API call if credentials
        are absent. Default: require ``api_key_env``."""
        if self.cfg.api_key_env:
            require_env(self.cfg.api_key_env)

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

    # -- the call ---------------------------------------------------------

    @abstractmethod
    async def _call(self, prompt: str, system_prompt: str,
                    tools: list[dict[str, Any]]) -> tuple[Any, list[ToolCall], str]:
        """Perform one provider API call.

        Returns ``(raw_response_dict, native_tool_calls, text)``. Must raise
        (anything) on failure; the runner classifies and retries.
        """

    async def complete(self, sample: Sample, run_index: int,
                       system_prompt: str) -> ProviderResult:
        tools = [t.model_dump(mode="json") for t in sample.tools]
        started = time.monotonic()
        raw, calls, text = await self._call(sample.prompt, system_prompt, tools)
        latency_ms = int((time.monotonic() - started) * 1000)
        if not calls and self.settings.allow_text_tool_call_fallback:
            calls = parse_tool_calls_from_text(text)
        return ProviderResult(
            model=self.cfg.name,
            provider=self.cfg.provider,
            sample_id=sample.id,
            run_index=run_index,
            raw_response=raw,
            tool_calls=calls,
            text=text,
            error=None,
            latency_ms=latency_ms,
        )


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

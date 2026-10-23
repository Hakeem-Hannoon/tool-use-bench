"""Pydantic models for samples, tool calls, provider results, and scoring."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator,
)

SCHEMA_VERSION = "2"

#: Categories of the generic suite. Other suites (e.g. Spotter) define their
#: own; a category is any lowercase snake_case label.
GENERIC_CATEGORIES = (
    "single_tool",
    "multi_tool",
    "parallel_tools",
    "no_tool_decoy",
    "argument_precision",
    "ordering_required",
    "tool_choice_under_ambiguity",
)

#: Tool protocols a model can be benchmarked under.
#:
#: ``native``          the provider's own tool/function-calling API.
#: ``spotter_router``  Spotter's JSON-router contract: no ``tools`` parameter,
#:                     the catalog lives in the system prompt, and the model
#:                     answers each step with one JSON object
#:                     ``{"kind": "tool", "tool": ..., "arguments": ...}`` or
#:                     ``{"kind": "reply", ...}``.
PROTOCOLS = ("native", "spotter_router")


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FunctionDef(BaseModel):
    """The ``function`` payload of an OpenAI-style tool definition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def _parameters_must_be_object_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v.get("type") != "object":
            raise ValueError("tool parameters schema must have type 'object'")
        if not isinstance(v.get("properties", {}), dict):
            raise ValueError("tool parameters 'properties' must be an object")
        return v


class ToolDef(BaseModel):
    """OpenAI-style tool definition — the canonical wire format for samples.

    Provider adapters translate this shape into each provider's native format.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "function"
    function: FunctionDef

    @field_validator("type")
    @classmethod
    def _type_must_be_function(cls, v: str) -> str:
        if v != "function":
            raise ValueError("only tools of type 'function' are supported")
        return v


class ToolCall(BaseModel):
    """A single tool invocation — expected or actually emitted by a model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ExpectedToolCall(ToolCall):
    """An expected call. ``alternatives`` lists other (name, arguments) pairs
    that satisfy the same expectation — e.g. any of several read tools, or
    two accepted argument shapes for one tool."""

    alternatives: list[ToolCall] = Field(default_factory=list)

    def variants(self) -> list[ToolCall]:
        return [self, *self.alternatives]


class ScoringConfig(BaseModel):
    """Per-sample switches controlling how strictly the scorer compares calls."""

    model_config = ConfigDict(extra="forbid")

    require_exact_tool_names: StrictBool = True
    require_exact_arguments: StrictBool = True
    allow_extra_tool_calls: StrictBool = False
    require_order: StrictBool = True
    allow_argument_coercion: StrictBool = False
    trim_string_whitespace: StrictBool = True
    #: Calls to these tools never count as extra, even when
    #: ``allow_extra_tool_calls`` is false (e.g. harmless reads before a write).
    allowed_extra_tools: list[str] = Field(default_factory=list)
    #: Validate every emitted call's arguments against the offered tool's JSON
    #: Schema; a violation fails the run (what server-side validation rejects).
    validate_arguments_schema: StrictBool = False


class MockToolResult(BaseModel):
    """A scripted result returned when the model calls a tool mid-episode.

    The first entry whose ``tool`` matches and whose ``when`` (a subset of the
    call's arguments, matchers allowed) matches is returned to the model.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    when: dict[str, Any] = Field(default_factory=dict)
    result: Any


class Sample(BaseModel):
    """One benchmark sample: a prompt, the tools offered, and the expected calls."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^\d{4}_[a-z0-9_]+$")
    category: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    difficulty: Difficulty
    prompt: str = Field(min_length=1)
    system_prompt: str = ""
    #: Full system prompt per protocol. When present for the model's protocol
    #: it is used verbatim (it already carries any tool catalog and output
    #: contract); otherwise ``system_prompt`` / the default prompt is used.
    system_prompt_by_protocol: dict[str, str] = Field(default_factory=dict)
    tools: list[ToolDef]
    expected_tool_calls: list[ExpectedToolCall]
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    #: Steps the model may take. 1 = a single response (the classic
    #: single-turn sample). >1 runs an episode: non-terminal calls get a
    #: scripted result from ``tool_results`` and the model continues.
    max_steps: int = Field(default=1, ge=1, le=12)
    #: Tools whose call ends the episode (writes, display cards).
    terminal_tools: list[str] = Field(default_factory=list)
    #: Keep going after a terminal call (compound requests). The model then
    #: gets a "proposal created" result and may call another tool or reply.
    continue_after_terminal: StrictBool = False
    tool_results: list[MockToolResult] = Field(default_factory=list)
    notes: str = ""

    @field_validator("tools")
    @classmethod
    def _tool_names_unique(cls, v: list[ToolDef]) -> list[ToolDef]:
        names = [t.function.name for t in v]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tool names in sample")
        return v

    @field_validator("system_prompt_by_protocol")
    @classmethod
    def _known_protocols(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(v) - set(PROTOCOLS))
        if unknown:
            raise ValueError(f"unknown protocol(s) {unknown}; known: {list(PROTOCOLS)}")
        return v

    @model_validator(mode="after")
    def _matchers_are_valid(self) -> "Sample":
        from .scoring import validate_matchers

        problems: list[str] = []
        for i, call in enumerate(self.expected_tool_calls):
            problems += [f"expected_tool_calls[{i}].{p}"
                         for p in validate_matchers(call.arguments)]
            for j, alt in enumerate(call.alternatives):
                problems += [f"expected_tool_calls[{i}].alternatives[{j}].{p}"
                             for p in validate_matchers(alt.arguments)]
        for i, mock in enumerate(self.tool_results):
            problems += [f"tool_results[{i}].when.{p}"
                         for p in validate_matchers(mock.when)]
        if problems:
            raise ValueError("invalid matchers: " + "; ".join(problems))
        return self

    def tool_names(self) -> set[str]:
        return {t.function.name for t in self.tools}

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return {t.function.name: t.function.parameters for t in self.tools}

    def validate_expected_calls_reference_known_tools(self) -> None:
        """Fail fast if the sample names a tool it doesn't offer."""
        offered = self.tool_names()
        references = (
            [("expected tool call", v.name) for c in self.expected_tool_calls
             for v in c.variants()]
            + [("allowed extra tool", n) for n in self.scoring.allowed_extra_tools]
            + [("terminal tool", n) for n in self.terminal_tools]
            + [("mock tool result", m.tool) for m in self.tool_results]
        )
        for kind, name in references:
            if name not in offered:
                raise ValueError(
                    f"sample {self.id!r}: {kind} {name!r} "
                    f"is not among the offered tools {sorted(offered)}"
                )


class Usage(BaseModel):
    """Token usage for one provider call or a whole episode."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    #: Input tokens served from the provider's prompt cache (a subset of
    #: ``input_tokens``), billed at the cached rate when one is configured.
    cached_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


class EpisodeStep(BaseModel):
    """One model response inside an episode."""

    model_config = ConfigDict(extra="forbid")

    tool_calls: list[ToolCall] = Field(default_factory=list)
    text: str = ""
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int = 0
    #: Router protocol only: what happened with this step's JSON envelope
    #: ("tool", "reply", "prose", "invalid_json", "unknown_tool").
    envelope: str | None = None
    raw_response: Any = None


class ProviderResult(BaseModel):
    """Normalized output of one episode, identical across providers."""

    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str
    sample_id: str
    run_index: int
    raw_response: Any = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    text: str = ""
    error: str | None = None
    timed_out: bool = False
    latency_ms: int = 0
    usage: Usage = Field(default_factory=Usage)
    steps: list[EpisodeStep] = Field(default_factory=list)
    #: The episode ended in a state the product shows as an error (e.g. the
    #: router never produced a parseable JSON envelope).
    protocol_error: str | None = None
    #: Router nudges the episode needed (invalid JSON / unknown tool retries)
    #: — production recovers from these, but each one costs a model call.
    recoveries: int = 0


class ArgumentError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    argument_path: str
    # missing_argument | extra_argument | unexpected_argument | wrong_value |
    # type_mismatch
    error_type: str
    expected: Any = None
    actual: Any = None


class SchemaError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    argument_path: str
    message: str


class WrongToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: str | None = None
    actual: str | None = None


class ScoringResult(BaseModel):
    """Deterministic verdict for one run."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    missing_tool_calls: list[str] = Field(default_factory=list)
    wrong_tool_calls: list[WrongToolCall] = Field(default_factory=list)
    argument_errors: list[ArgumentError] = Field(default_factory=list)
    extra_tool_calls: list[str] = Field(default_factory=list)
    ordering_errors: list[str] = Field(default_factory=list)
    schema_errors: list[SchemaError] = Field(default_factory=list)
    protocol_error: str | None = None
    reason: str = ""


class FailureType(str, Enum):
    """Primary failure classification for a failed run (one per run)."""

    MISSING_TOOL_CALL = "missing_tool_call"
    WRONG_TOOL_NAME = "wrong_tool_name"
    SCHEMA_VIOLATION = "schema_violation"
    WRONG_ARGUMENT = "wrong_argument"
    EXTRA_TOOL_CALL = "extra_tool_call"
    ORDERING_ERROR = "ordering_error"
    PROTOCOL_ERROR = "protocol_error"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"


class RunRecord(BaseModel):
    """One completed model/sample/run combination, as stored in checkpoints
    and in the ``runs`` array of ``outputs/results.json``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str
    provider: str
    #: Stable config identity of the model entry (provider/name[/protocol]...)
    #: — what results are grouped by, so two entries for the same model
    #: (different protocol, endpoint, or deployment) never merge.
    model_key: str = ""
    #: Display name from the config (defaults to the model name).
    label: str = ""
    protocol: str = "native"
    sample_id: str
    run_index: int
    success: bool
    expected_tool_calls: list[ExpectedToolCall]
    actual_tool_calls: list[ToolCall]
    raw_model_output: Any = None
    text_output: str = ""
    scoring_result: ScoringResult
    latency_ms: int = 0
    error: str | None = None
    timed_out: bool = False
    failure_type: str | None = None
    usage: Usage = Field(default_factory=Usage)
    #: Cost of the run in USD from the model's configured pricing, or None
    #: when no pricing is configured.
    cost_usd: float | None = None
    step_count: int = 0
    recoveries: int = 0
    steps: list[EpisodeStep] = Field(default_factory=list)

    @field_validator("expected_tool_calls", mode="before")
    @classmethod
    def _accept_plain_tool_calls(cls, v: Any) -> Any:
        """Plain ToolCall objects are valid expectations (no alternatives)."""
        if isinstance(v, list):
            return [c.model_dump() if isinstance(c, ToolCall)
                    and not isinstance(c, ExpectedToolCall) else c for c in v]
        return v

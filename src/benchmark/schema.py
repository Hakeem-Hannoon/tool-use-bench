"""Pydantic models for samples, tool calls, provider results, and scoring."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

SCHEMA_VERSION = "1"


class Category(str, Enum):
    SINGLE_TOOL = "single_tool"
    MULTI_TOOL = "multi_tool"
    PARALLEL_TOOLS = "parallel_tools"
    NO_TOOL_DECOY = "no_tool_decoy"
    ARGUMENT_PRECISION = "argument_precision"
    ORDERING_REQUIRED = "ordering_required"
    TOOL_CHOICE_UNDER_AMBIGUITY = "tool_choice_under_ambiguity"


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


class ScoringConfig(BaseModel):
    """Per-sample switches controlling how strictly the scorer compares calls."""

    model_config = ConfigDict(extra="forbid")

    require_exact_tool_names: StrictBool = True
    require_exact_arguments: StrictBool = True
    allow_extra_tool_calls: StrictBool = False
    require_order: StrictBool = True
    allow_argument_coercion: StrictBool = False
    trim_string_whitespace: StrictBool = True


class Sample(BaseModel):
    """One benchmark sample: a prompt, the tools offered, and the expected calls."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^\d{4}_[a-z0-9_]+$")
    category: Category
    difficulty: Difficulty
    prompt: str = Field(min_length=1)
    system_prompt: str = ""
    tools: list[ToolDef]
    expected_tool_calls: list[ToolCall]
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    notes: str = ""

    @field_validator("tools")
    @classmethod
    def _tool_names_unique(cls, v: list[ToolDef]) -> list[ToolDef]:
        names = [t.function.name for t in v]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tool names in sample")
        return v

    def validate_expected_calls_reference_known_tools(self) -> None:
        """Fail fast if an expected call names a tool the sample doesn't offer."""
        offered = {t.function.name for t in self.tools}
        for call in self.expected_tool_calls:
            if call.name not in offered:
                raise ValueError(
                    f"sample {self.id!r}: expected tool call {call.name!r} "
                    f"is not among the offered tools {sorted(offered)}"
                )


class ProviderResult(BaseModel):
    """Normalized output of one provider call, identical across providers."""

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


class ArgumentError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    argument_path: str
    error_type: str  # missing_argument | extra_argument | wrong_value | type_mismatch
    expected: Any = None
    actual: Any = None


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
    reason: str = ""


class FailureType(str, Enum):
    """Primary failure classification for a failed run (one per run)."""

    MISSING_TOOL_CALL = "missing_tool_call"
    WRONG_TOOL_NAME = "wrong_tool_name"
    WRONG_ARGUMENT = "wrong_argument"
    EXTRA_TOOL_CALL = "extra_tool_call"
    ORDERING_ERROR = "ordering_error"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"


class RunRecord(BaseModel):
    """One completed model/sample/run combination, as stored in checkpoints
    and in the ``runs`` array of ``outputs/results.json``."""

    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str
    sample_id: str
    run_index: int
    success: bool
    expected_tool_calls: list[ToolCall]
    actual_tool_calls: list[ToolCall]
    raw_model_output: Any = None
    text_output: str = ""
    scoring_result: ScoringResult
    latency_ms: int = 0
    error: str | None = None
    timed_out: bool = False
    failure_type: str | None = None

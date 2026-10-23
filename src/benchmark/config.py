"""Benchmark run configuration: loading, validation, hashing, redaction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schema import PROTOCOLS, Usage
from .utils.hashing import sha256_of_json
from .utils.json_utils import load_json_file

SUPPORTED_PROVIDERS = (
    "openai",
    "azure_openai",
    "anthropic",
    "gemini",
    "vertexai",
    "openrouter",
    "openai_compatible",
)


class Pricing(BaseModel):
    """USD per million tokens. ``cached_input_per_mtok`` applies to input
    tokens the provider served from its prompt cache (defaults to the full
    input rate when omitted)."""

    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cached_input_per_mtok: float | None = Field(default=None, ge=0)

    def cost(self, usage: Usage) -> float:
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        cached_rate = (self.cached_input_per_mtok
                       if self.cached_input_per_mtok is not None
                       else self.input_per_mtok)
        return ((usage.input_tokens - cached) * self.input_per_mtok
                + cached * cached_rate
                + usage.output_tokens * self.output_per_mtok) / 1_000_000


class ModelConfig(BaseModel):
    """One model/provider pair to benchmark."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1)
    provider: str
    #: Display name in reports; also distinguishes two entries that share a
    #: model (e.g. the same deployment at two temperatures).
    label: str | None = None
    #: Tool protocol: "native" (provider tool calling) or "spotter_router"
    #: (Spotter's JSON envelope; see benchmark.router).
    protocol: str = "native"
    api_key_env: str | None = None
    endpoint: str | None = None
    #: Read the endpoint from this environment variable instead (keeps
    #: resource URLs such as an Azure Foundry endpoint out of the config).
    endpoint_env: str | None = None
    pricing: Pricing | None = None
    # Azure-specific
    deployment: str | None = None
    api_version: str | None = None
    # Vertex AI-specific (credentials come from env var names, never values)
    project_id_env: str | None = None
    location_env: str | None = None
    service_account_path_env: str | None = None
    # Extra request parameters merged into every API call for this model
    api_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def _provider_supported(cls, v: str) -> str:
        if v not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"unsupported provider {v!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
            )
        return v

    @field_validator("protocol")
    @classmethod
    def _protocol_supported(cls, v: str) -> str:
        if v not in PROTOCOLS:
            raise ValueError(f"unsupported protocol {v!r}; supported: {', '.join(PROTOCOLS)}")
        return v

    @property
    def key(self) -> str:
        """Stable identifier for this model entry: provider + name, plus
        endpoint, deployment, a non-default protocol, and the label when set.
        Entries without the newer fields keep their old keys."""
        parts = [self.provider, self.name]
        if self.endpoint:
            parts.append(self.endpoint)
        elif self.endpoint_env:
            parts.append(f"${self.endpoint_env}")
        if self.deployment:
            parts.append(self.deployment)
        if self.protocol != "native":
            parts.append(self.protocol)
        key = "/".join(parts)
        return f"{key}#{self.label}" if self.label else key

    @property
    def display(self) -> str:
        if self.label:
            return self.label
        return self.name if self.protocol == "native" else f"{self.name} ({self.protocol})"


class BenchmarkConfig(BaseModel):
    """Top-level run configuration loaded from a JSON file."""

    model_config = ConfigDict(extra="forbid")

    benchmark: str = "benchmark_samples/full_benchmark.json"
    output: str = "outputs/results.json"
    report: str = "reports/tool_call_report.md"
    runs_per_sample: int = Field(default=5, ge=1)
    concurrency: int = Field(default=5, ge=1)
    temperature: float | None = 0
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=90, gt=0)
    allow_text_tool_call_fallback: bool = False
    resume: bool = True
    force: bool = False
    seed: int | None = 12345
    save_raw_responses: bool = True
    #: Success rate a model must reach on a category to be eligible as that
    #: workload's pick in the report's cost table.
    reliability_bar: float = Field(default=0.95, ge=0, le=1)
    models: list[ModelConfig] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def _model_keys_unique(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        keys = [m.key for m in v]
        if len(keys) != len(set(keys)):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"duplicate model entries in config: {dupes}")
        return v

    # ------------------------------------------------------------------
    # Loading / hashing / redaction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        data = load_json_file(path)
        if not isinstance(data, dict):
            raise ValueError(f"config file {path} must contain a JSON object")
        return cls.model_validate(data)

    def config_hash(self) -> str:
        """Deterministic hash of the config content (no secrets are ever in
        the config — only env var *names* — so hashing the full model dump
        is safe)."""
        return sha256_of_json(self.model_dump(mode="json"))

    def redacted_dump(self) -> dict[str, Any]:
        """Config snapshot for outputs.

        The config schema only ever stores environment variable *names*, never
        secret values, so redaction here is defense-in-depth: it asserts that
        shape and annotates it explicitly.
        """
        data = self.model_dump(mode="json")
        for model in data.get("models", []):
            for field in ("api_key_env", "endpoint_env", "project_id_env",
                          "location_env", "service_account_path_env"):
                if model.get(field):
                    # keep the env var NAME (not a secret) — value never loaded here
                    model[field] = str(model[field])
        data["_note"] = (
            "Secrets are read from environment variables at runtime and are "
            "never stored in this snapshot; only environment variable names appear."
        )
        return data

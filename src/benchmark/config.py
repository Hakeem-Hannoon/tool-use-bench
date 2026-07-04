"""Benchmark run configuration: loading, validation, hashing, redaction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class ModelConfig(BaseModel):
    """One model/provider pair to benchmark."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1)
    provider: str
    api_key_env: str | None = None
    endpoint: str | None = None
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

    @property
    def key(self) -> str:
        """Stable identifier for this model entry (provider + name [+ endpoint])."""
        parts = [self.provider, self.name]
        if self.endpoint:
            parts.append(self.endpoint)
        if self.deployment:
            parts.append(self.deployment)
        return "/".join(parts)


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
            for field in ("api_key_env", "project_id_env", "location_env",
                          "service_account_path_env"):
                if model.get(field):
                    # keep the env var NAME (not a secret) — value never loaded here
                    model[field] = str(model[field])
        data["_note"] = (
            "Secrets are read from environment variables at runtime and are "
            "never stored in this snapshot; only environment variable names appear."
        )
        return data

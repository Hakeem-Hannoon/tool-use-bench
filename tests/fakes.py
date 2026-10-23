"""Offline providers for tests: scripted step outputs, recorded history."""

from __future__ import annotations

import json
from typing import Any, Callable

from benchmark.config import ModelConfig
from benchmark.providers.base import (
    BaseProvider, Conversation, NativeCall, ProviderSettings, StepOutput,
)
from benchmark.schema import Usage

Script = Callable[[Conversation, int], StepOutput]


def native(*calls: tuple[str, dict[str, Any]], text: str = "",
           usage: Usage | None = None) -> StepOutput:
    """A step that emits native tool calls."""
    return StepOutput(
        raw={"scripted": True},
        calls=[NativeCall(id=f"call_{i}", name=n, arguments=a)
               for i, (n, a) in enumerate(calls)],
        text=text, usage=usage or Usage(input_tokens=100, output_tokens=10),
    )


def text(content: str, usage: Usage | None = None) -> StepOutput:
    """A step that only returns text (a router envelope or a reply)."""
    return StepOutput(raw={"scripted": True}, calls=[], text=content,
                      usage=usage or Usage(input_tokens=100, output_tokens=10))


def envelope(tool: str | None = None, reply: str | None = None,
             **arguments: Any) -> StepOutput:
    if reply is not None:
        return text(json.dumps({"kind": "reply", "message": reply}))
    return text(json.dumps({"kind": "tool", "tool": tool,
                            "arguments": arguments}))


class ScriptedProvider(BaseProvider):
    """Plays back ``script`` (a list of StepOutputs, or a callable taking the
    conversation and step index) and records every request it receives."""

    name = "scripted"

    def __init__(self, cfg: ModelConfig, settings: ProviderSettings,
                 script: list[StepOutput] | Script):
        super().__init__(cfg, settings)
        self.script = script
        self.requests: list[dict[str, Any]] = []

    def check_credentials(self) -> None:
        return

    def user_message(self, content: str) -> dict[str, Any]:
        return {"role": "user", "content": content}

    def assistant_message(self, step: StepOutput) -> dict[str, Any]:
        return {"role": "assistant", "content": step.text,
                "calls": [c.name for c in step.calls]}

    def tool_result_messages(self, step: StepOutput,
                             results: list[tuple[NativeCall, str]]) -> list[Any]:
        return [{"role": "tool", "name": c.name, "content": payload}
                for c, payload in results]

    async def step(self, conv: Conversation, tools, *, json_mode: bool = False):
        index = len(self.requests)
        self.requests.append({
            "system": conv.system,
            "messages": [dict(m) for m in conv.messages],
            "tools": [t["function"]["name"] for t in tools] if tools else None,
            "json_mode": json_mode,
        })
        if callable(self.script):
            return self.script(conv, index)
        return self.script[index]


def model_cfg(**overrides: Any) -> ModelConfig:
    data = {"name": "fake-model", "provider": "openai", "api_key_env": None}
    data.update(overrides)
    return ModelConfig.model_validate(data)

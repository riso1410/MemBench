from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import ModelConfig
from .token_count import estimate_message_tokens, estimate_tokens


@dataclass(frozen=True)
class ModelResult:
    content: str
    usage: dict[str, int]
    raw: dict[str, Any]


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]]) -> ModelResult:
        ...


class DryRunModel:
    name = "dry_run"

    def complete(self, messages: list[dict[str, str]]) -> ModelResult:
        prompt_tokens = estimate_message_tokens(messages)
        content = (
            "DRY RUN: no LLM is configured.\n\n"
            "This scaffold built the prompt, retrieved memory, and stopped before code generation.\n"
            "Set [model].provider = \"openai_compatible\" and [model].base_url to run a real model."
        )
        return ModelResult(
            content=content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": estimate_tokens(content),
                "total_tokens": prompt_tokens + estimate_tokens(content),
            },
            raw={"provider": self.name},
        )


class OpenAICompatibleModel:
    name = "openai_compatible"

    def __init__(self, config: ModelConfig):
        if not config.base_url:
            raise ValueError("[model].base_url is required for openai_compatible provider")
        self.config = config
        self.endpoint = _chat_completions_endpoint(config.base_url)

    def complete(self, messages: list[dict[str, str]]) -> ModelResult:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"model request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"model request failed: {exc}") from exc

        raw = json.loads(response_body)
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"model response did not match OpenAI chat format: {raw}") from exc
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        normalized_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", estimate_message_tokens(messages))),
            "completion_tokens": int(usage.get("completion_tokens", estimate_tokens(content))),
            "total_tokens": int(
                usage.get(
                    "total_tokens",
                    estimate_message_tokens(messages) + estimate_tokens(content),
                )
            ),
        }
        return ModelResult(content=content, usage=normalized_usage, raw=raw)


def build_model(config: ModelConfig) -> ChatModel:
    provider = config.provider.lower().strip()
    if provider in {"dry_run", "dry-run", "none"}:
        return DryRunModel()
    if provider in {"openai_compatible", "openai-compatible", "openai"}:
        return OpenAICompatibleModel(config)
    raise ValueError(f"unknown model provider: {config.provider}")


def _chat_completions_endpoint(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/v1/chat/completions"


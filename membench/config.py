from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "dry_run"
    model: str = "dry-run"
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: int = 120
    max_output_tokens: int = 4096
    temperature: float = 0.0
    input_cost_per_million_tokens: float = 0.0
    output_cost_per_million_tokens: float = 0.0


@dataclass(frozen=True)
class MemoryConfig:
    adapter: str = "none"
    corpus_root: str = "dataset/examples/memory_corpora"
    top_k: int = 8
    max_memory_tokens: int = 3000


@dataclass(frozen=True)
class AgentConfig:
    backend: str = "single_shot"  # or "claude_code" or "opencode"
    claude_cmd: str = "claude"
    claude_model: str = ""
    opencode_cmd: str = "opencode"
    opencode_model: str = ""  # opencode model id, "provider/model" form
    permission_mode: str = "bypassPermissions"
    env: dict[str, str] = field(default_factory=dict)  # extra env for the claude process (e.g. ANTHROPIC_BASE_URL)


@dataclass(frozen=True)
class RunConfig:
    output_dir: str = "runs/default"
    max_instances: int = 0
    include_retrieved_memory_in_output: bool = True


@dataclass(frozen=True)
class MemBenchConfig:
    model: ModelConfig = ModelConfig()
    memory: MemoryConfig = MemoryConfig()
    agent: AgentConfig = AgentConfig()
    run: RunConfig = RunConfig()


def load_config(path: str | Path) -> MemBenchConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> MemBenchConfig:
    model_raw = raw.get("model", {})
    memory_raw = raw.get("memory", {})
    agent_raw = raw.get("agent", {})
    run_raw = raw.get("run", {})
    if not isinstance(model_raw, dict):
        raise ValueError("[model] must be a TOML table")
    if not isinstance(memory_raw, dict):
        raise ValueError("[memory] must be a TOML table")
    if not isinstance(agent_raw, dict):
        raise ValueError("[agent] must be a TOML table")
    if not isinstance(run_raw, dict):
        raise ValueError("[run] must be a TOML table")

    model = ModelConfig(
        provider=str(model_raw.get("provider", ModelConfig.provider)),
        model=str(model_raw.get("model", ModelConfig.model)),
        base_url=str(model_raw.get("base_url", ModelConfig.base_url)),
        api_key_env=str(model_raw.get("api_key_env", ModelConfig.api_key_env)),
        timeout_sec=int(model_raw.get("timeout_sec", ModelConfig.timeout_sec)),
        max_output_tokens=int(model_raw.get("max_output_tokens", ModelConfig.max_output_tokens)),
        temperature=float(model_raw.get("temperature", ModelConfig.temperature)),
        input_cost_per_million_tokens=float(
            model_raw.get(
                "input_cost_per_million_tokens",
                ModelConfig.input_cost_per_million_tokens,
            )
        ),
        output_cost_per_million_tokens=float(
            model_raw.get(
                "output_cost_per_million_tokens",
                ModelConfig.output_cost_per_million_tokens,
            )
        ),
    )
    memory = MemoryConfig(
        adapter=str(memory_raw.get("adapter", MemoryConfig.adapter)),
        corpus_root=str(memory_raw.get("corpus_root", MemoryConfig.corpus_root)),
        top_k=int(memory_raw.get("top_k", MemoryConfig.top_k)),
        max_memory_tokens=int(memory_raw.get("max_memory_tokens", MemoryConfig.max_memory_tokens)),
    )
    agent = AgentConfig(
        backend=str(agent_raw.get("backend", AgentConfig.backend)),
        claude_cmd=str(agent_raw.get("claude_cmd", AgentConfig.claude_cmd)),
        claude_model=str(agent_raw.get("claude_model", AgentConfig.claude_model)),
        opencode_cmd=str(agent_raw.get("opencode_cmd", AgentConfig.opencode_cmd)),
        opencode_model=str(agent_raw.get("opencode_model", AgentConfig.opencode_model)),
        permission_mode=str(agent_raw.get("permission_mode", AgentConfig.permission_mode)),
        env={str(k): str(v) for k, v in dict(agent_raw.get("env", {})).items()},
    )
    run = RunConfig(
        output_dir=str(run_raw.get("output_dir", RunConfig.output_dir)),
        max_instances=int(run_raw.get("max_instances", RunConfig.max_instances)),
        include_retrieved_memory_in_output=bool(
            run_raw.get(
                "include_retrieved_memory_in_output",
                RunConfig.include_retrieved_memory_in_output,
            )
        ),
    )
    return MemBenchConfig(model=model, memory=memory, agent=agent, run=run)

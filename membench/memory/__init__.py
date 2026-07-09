from __future__ import annotations

from ..config import MemoryConfig
from .base import MemoryAdapter
from .none import NoMemoryAdapter
from .raw_rag import RawRagMemoryAdapter
from .structured import StructuredMemoryAdapter


def build_memory_adapter(config: MemoryConfig) -> MemoryAdapter:
    adapter = config.adapter.lower().strip()
    if adapter in {"none", "no_memory", "off"}:
        return NoMemoryAdapter()
    if adapter in {"raw_rag", "raw-history-rag"}:
        return RawRagMemoryAdapter(
            corpus_root=config.corpus_root,
            top_k=config.top_k,
            max_memory_tokens=config.max_memory_tokens,
        )
    if adapter in {"structured", "structured_project_memory"}:
        return StructuredMemoryAdapter(
            corpus_root=config.corpus_root,
            top_k=config.top_k,
            max_memory_tokens=config.max_memory_tokens,
        )
    if adapter in {"mem0", "mem0_oss"}:
        from .mem0_adapter import Mem0Adapter

        return Mem0Adapter(
            corpus_root=config.corpus_root,
            top_k=config.top_k,
            max_memory_tokens=config.max_memory_tokens,
        )
    if adapter in {"claude_mem", "claude-mem"}:
        from .claude_mem import ClaudeMemAdapter

        return ClaudeMemAdapter(
            corpus_root=config.corpus_root,
            top_k=config.top_k,
            max_memory_tokens=config.max_memory_tokens,
        )
    if adapter == "graphiti":
        from .graphiti_adapter import GraphitiAdapter

        return GraphitiAdapter(
            corpus_root=config.corpus_root,
            top_k=config.top_k,
            max_memory_tokens=config.max_memory_tokens,
        )
    if adapter == "graphify":
        from .graphify import GraphifyAdapter

        return GraphifyAdapter(
            corpus_root=config.corpus_root,
            top_k=config.top_k,
            max_memory_tokens=config.max_memory_tokens,
        )
    raise ValueError(f"unknown memory adapter: {config.adapter}")


__all__ = ["MemoryAdapter", "build_memory_adapter"]


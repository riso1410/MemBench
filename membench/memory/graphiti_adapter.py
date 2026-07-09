from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..jsonl import read_jsonl
from ..schema import memory_corpus_path
from .base import MemoryItem
from .mem0_adapter import LLM_BASE_URL, EMBED_BASE_URL, _corpus_texts
from .structured import _limit_tokens


class GraphitiAdapter:
    """Graphiti ("graphify") pinned to local models: qwen extraction into a temporal
    knowledge graph (embedded Kuzu), hybrid semantic+BM25 retrieval of graph facts.
    """

    name = "graphiti"

    def __init__(self, corpus_root: str | Path, top_k: int = 8, max_memory_tokens: int = 3000):
        self.corpus_root = Path(corpus_root)
        self.top_k = top_k
        self.max_memory_tokens = max_memory_tokens
        self.construction_time_sec: float = 0.0

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        corpus_dir = memory_corpus_path(instance, self.corpus_root)
        if not corpus_dir.exists():
            return []
        return asyncio.run(self._retrieve(corpus_dir, query))

    async def _retrieve(self, corpus_dir: Path, query: str) -> list[MemoryItem]:
        from graphiti_core import Graphiti
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.nodes import EpisodeType

        # graphiti's blacklist sanitize misses backticks etc. -> RediSearch syntax errors
        if not getattr(FalkorDriver, "_mb_sanitize_patched", False):
            _orig = FalkorDriver.sanitize
            FalkorDriver.sanitize = lambda self, q, _o=_orig: re.sub(r"[^0-9A-Za-z_\s]", " ", _o(self, q))
            FalkorDriver._mb_sanitize_patched = True

        store_dir = Path("runs/.graphiti_stores") / corpus_dir.name
        store_dir.mkdir(parents=True, exist_ok=True)
        ingested_marker = store_dir / ".ingested"

        llm_config = LLMConfig(
            api_key="dummy",
            model="qwen3-coder-30b",
            small_model="qwen3-coder-30b",
            base_url=LLM_BASE_URL,
        )
        # Kuzu backend is deprecated in graphiti-core and creates no FTS indexes;
        # FalkorDB (docker: membench-falkordb on :6379) is graphiti's supported default.
        graphiti = Graphiti(
            graph_driver=FalkorDriver(host="localhost", port=6379, database=corpus_dir.name),
            llm_client=OpenAIGenericClient(config=llm_config),
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key="dummy",
                    embedding_model="qwen3-embedding",
                    embedding_dim=1024,
                    base_url=EMBED_BASE_URL,
                )
            ),
            cross_encoder=OpenAIRerankerClient(config=llm_config),
        )
        try:
            started = time.time()
            if not ingested_marker.exists():
                await graphiti.build_indices_and_constraints()
                for index, (text, created_at) in enumerate(_corpus_texts(corpus_dir)):
                    reference_time = _parse_time(created_at)
                    await graphiti.add_episode(
                        name=f"{corpus_dir.name}_{index}",
                        episode_body=text,
                        source=EpisodeType.text,
                        source_description="project history",
                        reference_time=reference_time,
                    )
                ingested_marker.touch()
            self.construction_time_sec = round(time.time() - started, 3)

            safe_query = re.sub(r"[^\w\s]", " ", query)  # RediSearch chokes on raw issue text
            results = await graphiti.search(safe_query, num_results=self.top_k)
            items = [
                MemoryItem(
                    item_id=str(edge.uuid),
                    text=str(edge.fact),
                    source="graphiti",
                    score=0.0,
                    created_at=str(edge.valid_at or ""),
                    metadata={"construction_time_sec": self.construction_time_sec},
                )
                for edge in results
            ]
            return _limit_tokens(items, self.max_memory_tokens)
        finally:
            await graphiti.close()

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None


def _parse_time(created_at: str) -> datetime:
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)

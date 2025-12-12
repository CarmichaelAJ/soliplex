"""Query helper for the Soliplex/haiku-rag pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, load_yaml_config

PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PIPELINE_DIR / "haiku.rag.yaml"
DB_PATH = PIPELINE_DIR / "db" / "rag.lancedb"


@lru_cache(maxsize=1)
def _load_config() -> AppConfig:
    data = load_yaml_config(CONFIG_PATH)
    return AppConfig.model_validate(data)


async def _query_async(
    question: str,
    top_k: int,
    expand_context: bool,
) -> Dict[str, Any]:
    config = _load_config()
    async with HaikuRAG(db_path=DB_PATH, config=config) as rag:
        hits = await rag.search(question, limit=top_k, search_type="hybrid")
        if expand_context:
            hits = await rag.expand_context(hits)
        formatted = _format_results(hits)
        return {"query": question, "results": formatted}


def _format_results(hits: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for idx, (chunk, score) in enumerate(hits, start=1):
        formatted.append(
            {
                "rank": idx,
                "score": score,
                "document": chunk.content,
                "metadata": chunk.metadata,
                "document_info": {
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "document_uri": chunk.document_uri,
                    "document_meta": chunk.document_meta,
                },
            }
        )
    return formatted


def query_vector_db(
    question: str,
    top_k: int = 5,
    expand_context: bool = True,
) -> Dict[str, Any]:
    """Public synchronous helper for benchmarking scripts."""
    return asyncio.run(_query_async(question, top_k, expand_context))


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Query the Soliplex LanceDB database.")
    parser.add_argument("question", help="Question to search for.")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of hits to return.")
    parser.add_argument(
        "--no-expand",
        action="store_false",
        dest="expand_context",
        help="Disable context window expansion.",
    )
    args = parser.parse_args()
    result = query_vector_db(args.question, top_k=args.top_k, expand_context=args.expand_context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()

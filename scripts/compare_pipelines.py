#!/usr/bin/env python
"""Benchmark traditional HaikuRAG retrieval vs. the neighbor-aware selector.

The script runs the same set of natural-language questions through two
pipelines:

1. Baseline (`client.search` with the supplied limit)
2. Neighbor-aware selection (greedy utility maximization)

For each question we record latency, chunk counts, scores, and estimated
token usage. Results are printed to stdout and optionally exported to JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from haiku.rag import client as rag_client
from haiku.rag.config import AppConfig, load_yaml_config
from haiku.rag.store.models.chunk import Chunk

from soliplex.chunk_selection import (
    NeighborAwareSelectionConfig,
    apply_neighbor_aware_selection,
)


class TokenEstimator:
    """Approximate token counts without hard dependency on tiktoken."""

    def __init__(self) -> None:
        encoder = None
        try:  # pragma: no cover - optional dependency
            import tiktoken  # type: ignore

            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - best-effort fallback
            encoder = None
        self._encoder = encoder

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # crude fallback: whitespace tokenization
        return max(1, len(text.split()))


def load_questions(path: Path | None, inline: list[str]) -> list[str]:
    if inline:
        return [question.strip() for question in inline if question.strip()]
    if path is None:
        raise ValueError("Provide --question or --questions-file")
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        questions = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            questions.append(payload["question"])
        return questions
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("questions", [])
        return [item["question"] if isinstance(item, dict) else str(item) for item in payload]
    # default: treat as newline-delimited text file
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_neighbor_config(path: Path | None) -> NeighborAwareSelectionConfig | None:
    if path is None:
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return NeighborAwareSelectionConfig.from_mapping(data)


@dataclass
class PipelineMetrics:
    name: str
    duration_ms: float
    chunk_count: int
    token_count: int
    char_count: int
    avg_score: float | None
    max_score: float | None
    chunks: list[dict]


def summarize_hits(
    hits: Sequence[tuple[Chunk, float]],
    duration_s: float,
    estimator: TokenEstimator,
    label: str,
) -> PipelineMetrics:
    chunk_info = []
    token_total = 0
    char_total = 0
    scores = []
    for chunk, score in hits:
        chunk_info.append(
            {
                "document_id": chunk.document_id,
                "score": score,
                "uri": chunk.document_uri,
                "title": chunk.document_title,
            }
        )
        text = chunk.content or ""
        token_total += estimator.count(text)
        char_total += len(text)
        scores.append(score)

    avg_score = statistics.fmean(scores) if scores else None
    max_score = max(scores) if scores else None

    return PipelineMetrics(
        name=label,
        duration_ms=duration_s * 1000,
        chunk_count=len(hits),
        token_count=token_total,
        char_count=char_total,
        avg_score=avg_score,
        max_score=max_score,
        chunks=chunk_info,
    )


async def run_baseline(
    client: rag_client.HaikuRAG,
    question: str,
    limit: int,
    estimator: TokenEstimator,
) -> PipelineMetrics:
    start = time.perf_counter()
    hits = await client.search(question, limit=limit)
    elapsed = time.perf_counter() - start
    return summarize_hits(hits, elapsed, estimator, "baseline")


async def run_neighbor(
    client: rag_client.HaikuRAG,
    question: str,
    base_limit: int,
    estimator: TokenEstimator,
    selection_cfg: NeighborAwareSelectionConfig,
) -> PipelineMetrics:
    search_limit = selection_cfg.requested_candidates(base_limit)
    start = time.perf_counter()
    hits = await client.search(question, limit=search_limit)
    if not hits:
        elapsed = time.perf_counter() - start
        return summarize_hits([], elapsed, estimator, "neighbor")

    selected = await apply_neighbor_aware_selection(
        hits=list(hits),
        embedder=client.chunk_repository.embedder,
        config=selection_cfg,
        fallback_limit=base_limit,
    )
    elapsed = time.perf_counter() - start
    return summarize_hits(selected, elapsed, estimator, "neighbor")


def overlap(a: Iterable[dict], b: Iterable[dict]) -> int:
    ids_a = {item.get("document_id") for item in a if item.get("document_id")}
    ids_b = {item.get("document_id") for item in b if item.get("document_id")}
    return len(ids_a & ids_b)


def format_metrics(metrics: PipelineMetrics) -> str:
    avg = f"{metrics.avg_score:.4f}" if metrics.avg_score is not None else "n/a"
    max_score = f"{metrics.max_score:.4f}" if metrics.max_score is not None else "n/a"
    return (
        f"{metrics.name:9} | {metrics.duration_ms:8.2f} ms | "
        f"chunks={metrics.chunk_count:2d} | tokens={metrics.token_count:5d} | "
        f"avg={avg} | max={max_score}"
    )


async def evaluate(
    config_path: Path,
    db_path: Path,
    chunk_cfg: NeighborAwareSelectionConfig | None,
    questions: list[str],
    limit: int,
) -> dict:
    config_data = load_yaml_config(config_path)
    app_config = AppConfig.model_validate(config_data)

    estimator = TokenEstimator()
    results = []

    async with rag_client.HaikuRAG(db_path=db_path, config=app_config) as client:
        for idx, question in enumerate(questions, start=1):
            print(f"\n[{idx}/{len(questions)}] {question}")
            baseline = await run_baseline(client, question, limit, estimator)
            print(" ", format_metrics(baseline))
            if chunk_cfg is not None:
                neighbor = await run_neighbor(
                    client, question, limit, estimator, chunk_cfg
                )
                print(" ", format_metrics(neighbor))
            else:
                neighbor = None

            record = {
                "question": question,
                "baseline": asdict(baseline),
            }
            if neighbor:
                record["neighbor"] = asdict(neighbor)
                record["overlap"] = overlap(baseline.chunks, neighbor.chunks)
            results.append(record)

    return {"runs": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--haiku-config",
        type=Path,
        default=Path("example/haiku.rag.yaml"),
        help="YAML config used to connect to the LanceDB store",
    )
    parser.add_argument(
        "--rag-db",
        type=Path,
        default=Path("db/rag/rag.lancedb"),
        help="Path to the LanceDB directory",
    )
    parser.add_argument(
        "--chunk-config",
        type=Path,
        default=Path("config/chunk_selection.yaml"),
        help="Neighbor-aware chunk selection YAML. Omit to disable comparison.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="Path to a .txt/.json/.jsonl file with questions",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Inline question (may be supplied multiple times)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Top-K limit for the baseline search",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file to write the raw metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        questions = load_questions(args.questions_file, args.question)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        return 1

    chunk_cfg = None
    if args.chunk_config:
        chunk_cfg = load_neighbor_config(args.chunk_config)

    payload = asyncio.run(
        evaluate(
            config_path=args.haiku_config,
            db_path=args.rag_db,
            chunk_cfg=chunk_cfg,
            questions=questions,
            limit=args.limit,
        )
    )

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

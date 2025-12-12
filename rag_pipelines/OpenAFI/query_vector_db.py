"""Query helper for the OpenAFI/Chroma pipeline."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional

from .Vector_DB_1536 import (
    EMBEDDING_MODEL,
    OpenAIEmbeddingFunction,
    collection,
)

ACCURACY_THRESHOLD = 72
MAX_RESULTS = 7
CONTEXT_OFFSETS = (-1, 1)
NEIGHBOR_THRESHOLD = max(
    1, int(round(ACCURACY_THRESHOLD / max(1, len(CONTEXT_OFFSETS))))
)


def _calculate_accuracy(distance: float) -> int:
    base_accuracy = (1 - distance) * 100
    weighted = min(100 - (100 - base_accuracy) * 0.35, 98)
    return int(round(weighted))


def _fetch_neighbor(source_pdf: Optional[str], chunk_index: Optional[int]) -> Optional[str]:
    if not source_pdf or chunk_index is None:
        return None
    try:
        res = collection.get(
            where={"source_pdf": source_pdf, "chunk_index": chunk_index},
            include=["documents"],
            limit=1,
        )
        docs = res.get("documents") or []
        if docs and docs[0]:
            doc = docs[0][0]
            if isinstance(doc, str):
                return doc.strip()
    except Exception:
        return None
    return None


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    denom_a = math.sqrt(sum(a * a for a in vec_a))
    denom_b = math.sqrt(sum(b * b for b in vec_b))
    denom = denom_a * denom_b
    if denom == 0:
        return 0.0
    return max(min(numerator / denom, 1.0), -1.0)


def _score_neighbor(
    neighbor_text: Optional[str],
    query_embedding: Optional[List[float]],
    embedding_function: Optional[OpenAIEmbeddingFunction],
) -> Optional[int]:
    if not neighbor_text or not query_embedding or embedding_function is None:
        return None
    try:
        embedding = embedding_function(neighbor_text)
        if not embedding:
            return None
        similarity = _cosine_similarity(query_embedding, embedding[0])
        accuracy = max(0, int(round(similarity * 100)))
        return accuracy
    except Exception:
        return None


def _augment_with_context(
    entry: Dict[str, Any],
    query_embedding: Optional[List[float]],
    embedding_function: Optional[OpenAIEmbeddingFunction],
) -> None:
    meta = entry.get("metadata") or {}
    source = meta.get("source_pdf")
    chunk_idx = meta.get("chunk_index")
    if isinstance(chunk_idx, str) and chunk_idx.isdigit():
        chunk_idx = int(chunk_idx)
    if not isinstance(chunk_idx, int):
        return

    neighbor_details: List[Dict[str, Any]] = []
    left_segments: List[str] = []
    right_segments: List[str] = []

    for offset in CONTEXT_OFFSETS:
        neighbor_text = _fetch_neighbor(source, chunk_idx + offset)
        accuracy = _score_neighbor(neighbor_text, query_embedding, embedding_function)
        required_accuracy = NEIGHBOR_THRESHOLD
        include = (
            isinstance(neighbor_text, str)
            and neighbor_text.strip()
            and accuracy is not None
            and accuracy >= required_accuracy
        )

        neighbor_details.append(
            {
                "offset": offset,
                "included": include,
                "required_accuracy": required_accuracy,
                "accuracy": accuracy,
            }
        )

        if include:
            cleaned = neighbor_text.strip()
            if offset < 0:
                left_segments.append(cleaned)
            else:
                right_segments.append(cleaned)

    segments = left_segments + [entry.get("document")] + right_segments
    combined = "\n\n".join(
        seg for seg in segments if isinstance(seg, str) and seg.strip()
    )
    if combined:
        entry["document"] = combined
    if neighbor_details:
        entry["neighbor_evaluations"] = neighbor_details


def _format_results(
    question: str,
    docs: List[str],
    metadatas: List[Dict[str, Any]],
    distances: List[float],
    query_embedding: Optional[List[float]],
    embedding_function: Optional[OpenAIEmbeddingFunction],
) -> Dict[str, Any]:
    filtered: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []

    for doc, meta, distance in zip(docs, metadatas, distances):
        accuracy = _calculate_accuracy(distance)
        entry = {
            "rank": len(fallback) + 1,
            "distance": distance,
            "accuracy": accuracy,
            "document": doc,
            "metadata": meta,
            "chunk_info": {
                "source_pdf": meta.get("source_pdf"),
                "chunk_index": meta.get("chunk_index"),
                "token_count": meta.get("token_count"),
                "paragraph_range": meta.get("paragraph_range"),
            },
        }
        fallback.append(entry)
        if accuracy >= ACCURACY_THRESHOLD:
            filtered.append(entry)

    candidates = (filtered or fallback)[:MAX_RESULTS]

    for entry in candidates:
        _augment_with_context(entry, query_embedding, embedding_function)

    for idx, entry in enumerate(candidates, start=1):
        entry["rank"] = idx

    return {"query": question, "results": candidates}



def query_vector_db(question: str, top_k: int = 5, debug: bool = False) -> Dict[str, Any]:
    """Query the Chroma database for a single question."""
    if debug:
        print(f"[OpenAFI] Querying: {question!r}")
    embedding_function = OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try:
        query_embedding_list = embedding_function.embed_query(question)
        query_embedding = query_embedding_list[0] if query_embedding_list else None
    except Exception:
        query_embedding = None

    response = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    return _format_results(
        question,
        docs,
        metadatas,
        distances,
        query_embedding,
        embedding_function,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Query the OpenAFI Chroma database.")
    parser.add_argument("question", help="Question to search for.")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of hits to return.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    result = query_vector_db(args.question, top_k=args.top_k, debug=args.debug)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()

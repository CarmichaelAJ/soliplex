"""Compare retrieval quality between the OpenAFI and Soliplex pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rag_pipelines.OpenAFI import query_vector_db as openafi_query
from rag_pipelines.soliplex import query_lancedb

PIPELINES = {
    "OpenAFI": openafi_query,
    "Soliplex": query_lancedb,
}


def load_questions(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError("Questions file must be a list or a mapping with a 'questions' key.")

    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise ValueError("Questions must be a list of strings.")
    return questions


def benchmark(questions: List[str], top_k: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for question in questions:
        entry: Dict[str, Any] = {"question": question, "responses": {}}
        for name, query_fn in PIPELINES.items():
            try:
                entry["responses"][name] = query_fn(question, top_k=top_k)
            except Exception as exc:  # pragma: no cover - informational
                entry["responses"][name] = {
                    "query": question,
                    "error": str(exc),
                }
        results.append(entry)
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    for entry in results:
        print(f"\nQuestion: {entry['question']}")
        for name, payload in entry["responses"].items():
            if "error" in payload:
                print(f"  - {name}: ERROR -> {payload['error']}")
                continue
            hits = payload.get("results", [])
            if not hits:
                print(f"  - {name}: no hits")
                continue
            top_hit = hits[0]
            snippet = top_hit["document"][:120].replace("\n", " ")
            print(
                f"  - {name}: top score={top_hit.get('score') or top_hit.get('distance')} "
                f"chunk={top_hit['metadata'] or top_hit.get('document_info')} "
                f"text='{snippet}...'"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark both vector databases.")
    parser.add_argument(
        "--questions",
        default=Path(__file__).with_name("questions.json"),
        type=Path,
        help="Path to the questions JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to store the benchmark results as JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of hits to request from each vector DB.",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    results = benchmark(questions, top_k=args.top_k)
    print_summary(results)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()

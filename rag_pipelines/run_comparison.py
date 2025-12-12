"""Generate plain-text comparison of answers retrieved from each vector DB."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_pipelines.OpenAFI.query_vector_db import query_vector_db as openafi_query
from rag_pipelines.soliplex.query_lancedb import query_vector_db as soliplex_query

PIPELINES = {
    "OpenAFI": openafi_query,
    "Soliplex": soliplex_query,
}


def load_questions(path: Path) -> List[Dict[str, Optional[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "questions" in raw:
        raw = raw["questions"]
    questions: List[Dict[str, Optional[str]]] = []
    if not isinstance(raw, list):
        raise ValueError("Questions file must contain a list or a mapping with 'questions'.")

    for entry in raw:
        if isinstance(entry, str):
            questions.append({"question": entry, "reference": None})
        elif isinstance(entry, dict):
            q = entry.get("question")
            if not isinstance(q, str):
                raise ValueError("Each question entry must include a string 'question'.")
            questions.append({"question": q, "reference": entry.get("reference")})
        else:
            raise ValueError("Unsupported question entry type.")
    return questions


def aggregate_answer(payload: Dict[str, Any], combine: int) -> str:
    hits = payload.get("results") or []
    if not hits:
        return ""
    snippets = []
    for hit in hits[:combine]:
        text = hit.get("document", "")
        if text:
            snippets.append(text.strip())
    return "\n\n".join(snippets)


def similarity(candidate: str, reference: Optional[str]) -> Optional[float]:
    if not candidate or not reference:
        return None
    return SequenceMatcher(None, candidate.lower(), reference.lower()).ratio()


def generate_report(
    questions: List[Dict[str, Optional[str]]],
    top_k: int,
    combine: int,
) -> List[str]:
    report_lines: List[str] = []
    summary: List[str] = []

    for idx, entry in enumerate(questions, start=1):
        question = entry["question"]  # type: ignore[assignment]
        reference = entry.get("reference")
        report_lines.append(f"Question {idx}: {question}")
        if reference:
            report_lines.append(f"Reference Answer: {reference}")
        for name, query_fn in PIPELINES.items():
            try:
                payload = query_fn(question, top_k=top_k)
            except Exception as exc:  # pragma: no cover - logging only
                report_lines.append(f"  [{name}] ERROR: {exc}")
                summary.append(f"{name}\t{idx}\tERROR")
                continue

            answer = aggregate_answer(payload, combine=combine)
            score = similarity(answer, reference)
            report_lines.append(f"  [{name}] Retrieved Score: {score if score is not None else 'n/a'}")
            if answer:
                excerpt = answer.replace("\n", " ").strip()
                report_lines.append(f"  [{name}] Retrieved Text: {excerpt}")
            else:
                report_lines.append(f"  [{name}] Retrieved Text: <no results>")
            summary.append(f"{name}\t{idx}\t{score if score is not None else 'n/a'}")
        report_lines.append("-" * 80)

    report_lines.append("Summary (pipeline-question-score):")
    report_lines.extend(summary)
    return report_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare answers across vector DBs.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("questions.json"),
        help="Path to questions JSON (list or {questions: list}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("comparison_results.txt"),
        help="Output text file for the report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of hits to request from each retrieval pipeline.",
    )
    parser.add_argument(
        "--combine",
        type=int,
        default=1,
        help="Number of top chunks to concatenate into the candidate answer.",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    report = generate_report(questions, top_k=args.top_k, combine=args.combine)
    args.output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote comparison results to {args.output}")


if __name__ == "__main__":
    main()

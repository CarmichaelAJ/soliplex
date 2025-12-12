#!/usr/bin/env python
"""Run baseline HaikuRAG evaluations plus neighbor-aware telemetry in one go."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def default_paths() -> dict[str, Path]:
    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    workspace_root = repo_root.parent
    haiku_repo = workspace_root / "haiku.rag"
    cli_dir = haiku_repo / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    cli_name = "evaluations.exe" if os.name == "nt" else "evaluations"
    cli_path = cli_dir / cli_name
    return {
        "repo_root": repo_root,
        "haiku_repo": haiku_repo,
        "config": repo_root / "rag_pipelines" / "soliplex" / "haiku.rag.yaml",
        "db": repo_root / "rag_pipelines" / "soliplex" / "db" / "rag.lancedb",
        "chunk_config": repo_root / "config" / "chunk_selection.yaml",
        "logs": repo_root / "logs" / "evaluations",
        "evaluations_cli": cli_path,
    }


def run_command(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        header = f"$ {' '.join(cmd)}\n\n"
        stream.write(header)
        stream.flush()
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
        process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {' '.join(cmd)}"
        )


def extract_section(log_path: Path, header_prefix: str) -> str | None:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(header_prefix):
            start = idx
            break
    if start is None:
        return None
    section: list[str] = []
    for line in lines[start:]:
        if (
            line.startswith("===")
            and section
            and not line.startswith(header_prefix)
            and line.strip() != header_prefix.strip()
        ):
            break
        section.append(line.rstrip())
    return "\n".join(section).strip()


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="repliqa",
        help="Evaluation dataset key (as defined in haiku.rag evaluations).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit the number of QA / retrieval cases per run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=paths["config"],
        help="haiku.rag YAML config passed to the baseline CLI.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=paths["db"],
        help="LanceDB directory passed to the baseline CLI.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        required=True,
        help="Question list used for the neighbor-aware telemetry run "
        "(txt/json/jsonl accepted by compare_pipelines.py).",
    )
    parser.add_argument(
        "--chunk-config",
        type=Path,
        default=paths["chunk_config"],
        help="Neighbor-aware chunk selection YAML.",
    )
    parser.add_argument(
        "--haiku-path",
        type=Path,
        default=paths["haiku_repo"],
        help="Location of the haiku.rag repository (used for reference files).",
    )
    parser.add_argument(
        "--evaluations-cli",
        default=(paths["evaluations_cli"] if paths["evaluations_cli"].exists() else "evaluations"),
        help="Executable name for the HaikuRAG evaluations CLI.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Pass --skip-db to the evaluations CLI to reuse the existing LanceDB database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths["logs"],
        help="Directory where log files and JSON manifests are stored.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = default_paths()
    repo_root = paths["repo_root"]
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    manifest: dict[str, dict[str, str]] = {}
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # ------------------------------------------------------------------
    # Baseline: HaikuRAG evaluations CLI (retrieval + QA judges)
    # ------------------------------------------------------------------
    baseline_log = args.output_dir / f"{now}_baseline.log"
    baseline_cmd = [
        str(args.evaluations_cli),
        args.dataset,
        "--config",
        str(args.config),
        "--db",
        str(args.db),
        "--limit",
        str(args.limit),
    ]
    if args.skip_db:
        baseline_cmd.append("--skip-db")
    run_command(baseline_cmd, baseline_log, cwd=repo_root, env=env)
    manifest["baseline"] = {
        "command": " ".join(baseline_cmd),
        "log": str(baseline_log),
    }
    retrieval_summary = extract_section(
        baseline_log, "=== Retrieval Benchmark Results ==="
    )
    qa_summary = extract_section(baseline_log, "=== QA Benchmark Results ===")
    if retrieval_summary:
        manifest["baseline"]["retrieval_summary"] = retrieval_summary
    if qa_summary:
        manifest["baseline"]["qa_summary"] = qa_summary

    # ------------------------------------------------------------------
    # Neighbor-aware metrics: Soliplex compare_pipelines script
    # ------------------------------------------------------------------
    neighbor_log = args.output_dir / f"{now}_neighbor.log"
    neighbor_json = args.output_dir / f"{now}_neighbor_metrics.json"
    neighbor_cmd = [
        sys.executable,
        "scripts/compare_pipelines.py",
        "--questions-file",
        str(args.questions_file),
        "--haiku-config",
        str(args.config),
        "--rag-db",
        str(args.db),
        "--chunk-config",
        str(args.chunk_config),
        "--limit",
        str(args.limit),
        "--output",
        str(neighbor_json),
    ]
    run_command(neighbor_cmd, neighbor_log, cwd=repo_root, env=env)
    manifest["neighbor"] = {
        "command": " ".join(neighbor_cmd),
        "log": str(neighbor_log),
        "metrics": str(neighbor_json),
    }

    manifest_path = args.output_dir / f"{now}_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote evaluation manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

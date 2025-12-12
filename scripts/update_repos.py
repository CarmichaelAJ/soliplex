#!/usr/bin/env python
"""Utility script to sync the sibling Soliplex and HaikuRAG repos.

By default the script assumes this file lives inside the Soliplex repo and
that `../haiku.rag` is checked out next to it. Point the --haiku-path and
--soliplex-path flags elsewhere if your layout differs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RepoResult:
    name: str
    path: Path
    branch: str | None = None
    status_lines: list[str] | None = None


def _run_git(args: Iterable[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )


def update_repo(path: Path, pull: bool) -> RepoResult:
    result = RepoResult(name=path.name, path=path)
    if not path.exists():
        raise FileNotFoundError(path)

    fetch_proc = _run_git(["fetch", "--all", "--prune"], cwd=path)
    if fetch_proc.returncode != 0:
        raise RuntimeError(f"git fetch failed in {path}:\n{fetch_proc.stderr}")

    if pull:
        pull_proc = _run_git(["pull", "--ff-only"], cwd=path)
        if pull_proc.returncode != 0:
            raise RuntimeError(f"git pull failed in {path}:\n{pull_proc.stderr}")

    branch_proc = _run_git(
        ["rev-parse", "--abbrev-ref", "HEAD"],
        cwd=path,
    )
    if branch_proc.returncode == 0:
        result.branch = branch_proc.stdout.strip()

    status_proc = _run_git(["status", "-sb"], cwd=path)
    if status_proc.returncode == 0:
        result.status_lines = status_proc.stdout.strip().splitlines()
    else:
        result.status_lines = [status_proc.stderr.strip()]

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soliplex-path",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the Soliplex repository (default: repo root)",
    )
    parser.add_argument(
        "--haiku-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "haiku.rag",
        help="Path to the HaikuRAG repository (default: ../haiku.rag)",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Run `git pull --ff-only` after fetching",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repos = []

    for repo_path in (args.soliplex_path, args.haiku_path):
        try:
            repos.append(update_repo(repo_path, pull=args.pull))
        except Exception as exc:  # noqa: BLE001 - CLI surface
            print(f"[error] {exc}", file=sys.stderr)
            return 1

    for repo in repos:
        print(f"\n=== {repo.name} [{repo.path}] ===")
        branch = repo.branch or "unknown"
        print(f"branch: {branch}")
        if repo.status_lines:
            for line in repo.status_lines:
                print(f"  {line}")
        else:
            print("  (no status output)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

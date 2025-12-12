"""Ingest the Soliplex documentation into the default LanceDB database."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    pipeline_dir = Path(__file__).resolve().parent
    repo_root = pipeline_dir.parent.parent
    config_path = pipeline_dir / "haiku.rag.yaml"
    docs_dir = repo_root / "docs"
    db_path = pipeline_dir / "db" / "rag.lancedb"

    haiku_cli = shutil.which("haiku-rag")
    if haiku_cli is None:
        raise SystemExit(
            "haiku-rag CLI not found. Activate the venv and ensure the package is installed."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        haiku_cli,
        "--config",
        str(config_path),
        "add-src",
        "--db",
        str(db_path),
        str(docs_dir),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc

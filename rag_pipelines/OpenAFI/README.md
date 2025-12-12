# OpenAFI ChromaDB Pipeline

This folder mirrors the OpenAFI semantic ingestion workflow but points it at
the Soliplex documentation set (`docs/`). It builds a ChromaDB collection using
OpenAI embeddings and the paragraph chunking heuristics from `Vector_DB_1536.py`.

## Files

- `Vector_DB_1536.py` – ingestion script copied from the OpenAFI project with
  adjustments for the Soliplex docs:
  - Reads Markdown (and optional PDF/TXT) sources from the repo `docs/` folder.
  - Stores cached text outputs under `rag_pipelines/OpenAFI/text_cache/`.
  - Persists metadata trackers and stats alongside the script.
  - Writes a Chroma database to `rag_pipelines/OpenAFI/chroma_storage_1536/`.

## Requirements

Run inside the project virtual environment so dependencies like `chromadb`,
`pdfplumber`, and `openai` are available. Export `OPENAI_API_KEY` before
executing so embeddings can be generated.

## Usage

```bash
source .venv/bin/activate        # or .\.venv\Scripts\activate on PowerShell
export OPENAI_API_KEY=sk-...     # required for embedding calls
python rag_pipelines/OpenAFI/Vector_DB_1536.py
```

The script will:

1. Enumerate Markdown/TXT/PDF sources in `docs/`.
2. Rechunk and ingest anything new/changed into the Chroma collection.
3. Remove entries from the collection when their source file is deleted.

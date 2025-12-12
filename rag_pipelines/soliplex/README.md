# Soliplex Default RAG Pipeline

This folder is an exact copy of the ingestion pipeline described in the
project `README` and `docs/rag.md`.

- `haiku.rag.yaml` is copied verbatim from `example/haiku.rag.yaml`.
- `ingest_docs.py` runs the same command listed in the docs, but stores the
  LanceDB output under this folder (`rag_pipelines/soliplex/db/rag.lancedb`).
- `chunk_selection.yaml` mirrors the neighbor-aware selector settings published
  under `config/chunk_selection.yaml` so evaluation scripts can point at a local
  copy that lives next to the pipeline.

## Usage

1. Activate the project virtual environment.
2. Ensure `haiku-rag` is on your `PATH` (installed automatically with `pip install -e .`).
3. Run the ingestion script from the repo root:

   ```bash
   python rag_pipelines/soliplex/ingest_docs.py
   ```

The script builds `rag_pipelines/soliplex/db/rag.lancedb` from the Markdown
files in `docs/` using the copied `haiku.rag.yaml` settings, so each pipeline
variation stays self-contained.

When running evaluation suites (see `scripts/run_eval_suite.py`) you can pass:

```bash
python scripts/run_eval_suite.py \
  --config rag_pipelines/soliplex/haiku.rag.yaml \
  --db rag_pipelines/soliplex/db/rag.lancedb \
  --chunk-config rag_pipelines/soliplex/chunk_selection.yaml \
  --questions-file ..\repliqa-limit50.txt
```

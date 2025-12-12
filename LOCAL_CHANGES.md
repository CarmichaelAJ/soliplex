# Local Changes

## 2025-12-12

- Added `.env.example` plus gitignore hardening so GPT-OSS OpenAI keys stay local; README and agent docs now walk through copying the template and wiring `OPENAI_API_KEY`.
- Switched the default haiku/installation configs to hit `o4-mini-2025-04-16` via the OpenAI provider, aligning both HaikuRAG and Soliplex rooms on the same cloud model.
- Documented the neighbor-aware chunk selector inside `docs/config/rooms.md` and introduced dedicated scripts: `scripts/update_repos.py` keeps the paired `haiku.rag` + `soliplex` repos in sync, while `scripts/compare_pipelines.py` benchmarks baseline vs. neighbor-aware retrieval with token/time telemetry.
- Restored the `rag_pipelines/soliplex` example pipeline (config + LanceDB seed) so evaluation commands can reference `rag_pipelines/soliplex/haiku.rag.yaml` and `db/rag.lancedb`; the pipeline also includes the shared `chunk_selection.yaml`.
- Added `scripts/run_eval_suite.py`, which wraps the HaikuRAG `evaluations` CLI and the Soliplex neighbor-aware metrics run, streaming their output live while writing per-run logs, summaries, and JSON manifests to `logs/evaluations/`. Updated `docs/evaluations.md` and `README.md` with usage instructions.

## 2025-12-08

- Loosened the FastAPI `CORSMiddleware` settings (via env vars) and defaulted to allowing `localhost`/`127.0.0.1` origins so the Flutter web client can access the API without CORS failures.
- Updated the example installation at `example/minimal.yaml` so both `default_chat` and `alternate_chat` agents point at the locally installed `qwen3:4b` Ollama model, ensuring quiz grading works without hitting "model not found / unsupported tools" errors.
- Added a lenient AG-UI stream adapter (`src/flutter/lib/infrastructure/lenient_stream_adapter.dart`) and wired it into `PydanticProviderController` so malformed/experimental AG-UI events coming from the research room are logged and skipped instead of crashing the Flutter client with `LlmFailureException` decoding errors.
- Pinned the Haiku RAG configuration (`example/haiku.rag.yaml`) so both the QA and research agents run through the local `qwen3:4b` Ollama model; research runs now stay on-device instead of failing with `model 'gpt-oss' not found`.
- Pointed the Joker room's selector agent (`soliplex.examples.joker_agent_factory`) at the local `qwen3:4b` Ollama model instead of `qwen3:latest`, eliminating 404 errors when only 4b is installed.
- Left the Joker room's joke generator on the local `gpt-oss-latest` Ollama model so it keeps working offline even though the rest of the install now favors `o4-mini-2025-04-16`.
- May need option to check for models, download models, delete (?)

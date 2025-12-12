# Neighbor-Aware Chunk Selection – Progress Notes

## What changed
- Added a reusable selector (`src/soliplex/chunk_selection.py`) that:
  - Turns raw search scores into neighbor-reinforced pseudo-probabilities.
  - Penalizes redundant chunks via cosine similarity.
  - Greedily adds chunks until marginal utility drops below zero, yielding a dynamic top‑k.
- Extended `SearchDocumentsToolConfig` so rooms can enable the selector via a new `chunk_selection` block (no config = legacy top‑k).
- Updated `soliplex.tools.search_documents` and the Soliplex benchmark CLI to over-fetch when the selector is enabled and run the optimizer automatically.
- Documented the new YAML stanza in `docs/config/rooms.md` and added unit tests for both the selector and config parsing.
- Moved the public selector profile to `config/chunk_selection.yaml` so other tools (e.g., the `evaluations` CLI) can share it.

## Early results (RepliQA, limit 10)
| Retrieval mode          | QA Accuracy | Notes |
|-------------------------|-------------|-------|
| Baseline top‑k (legacy) | 6 / 10 (60%)| Standard `search_documents_limit = 5`, no reranker. |
| Neighbor-aware selector | 8 / 9 (88.9%) | Same Ollama stack (`qwen3-embedding:4b`, `qwen3-vl:8b`); final case aborted after the judge hit the retry cap, but the rest succeeded. |

Retrieval MRR stayed at 1.000 in both runs, so the gains came entirely from better context fed to the QA agent.

## Next steps
- Rebuild the LanceDB under `db/rag/haiku.rag.lancedb` and run 50-question sweeps (legacy vs neighbor-aware) so we can share a larger sample size.
- Measure the extra embedding cost of the selector and explore reusing stored vectors to reduce latency/billing.
- Push the feature branch to `github.com/CarmichaelAJ/soliplex` and open a PR once the 50/50 runs are recorded.

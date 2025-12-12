# Evaluation Workflow

This repository now treats `soliplex-main/` and the sibling `../haiku.rag`
checkout as two cooperating projects. The following flow keeps both trees in
sync, runs retrieval comparisons, and captures metrics for future regression
tests.

## 1. Keep the repos aligned

```bash
cd soliplex-main
python scripts/update_repos.py --pull
```

The script fetches/pulls both repos (paths are configurable) and prints the
current branch plus `git status -sb` for each. Run it before collecting data
or cutting a release branch so both HaikuRAG and Soliplex changes stay in lock
step.

## 2. Configure secrets and models

1. Copy `.env.example` → `.env` in both repositories.
2. Supply `OPENAI_API_KEY` for GPT-OSS:20b in each `.env`.
3. Point the LanceDB path referenced by your room configs at the corpus you
   plan to evaluate; rebuilding index data in `db/rag/` is recommended.

Both Haiku configs (`example/haiku.rag.yaml` here and `haiku.rag/haiku.rag.yaml`)
now default to GPT-OSS:20b via the OpenAI provider, so no further YAML edits
are required to use the shared key.

## 3. Compare retrieval strategies

Use the new `scripts/compare_pipelines.py` runner to benchmark the
traditional Haiku search (`top_k`) against the neighbor-aware selector:

```bash
python scripts/compare_pipelines.py \
  --questions-file ../tests/questions.jsonl \
  --rag-db db/rag/rag.lancedb \
  --chunk-config config/chunk_selection.yaml \
  --output logs/neighbor_eval.json
```

Key metrics per question:

- wall-clock time for each pipeline
- chunks returned, average score, token + character counts
- overlap between the two chunk sets

The script streams progress to stdout and, when `--output` is provided, stores
the raw telemetry so you can diff regressions over time.

For repeatable CLI-style orchestration run:

```bash
python scripts/run_eval_suite.py \
  --questions-file ..\repliqa-limit50.txt \
  --limit 50
```

This wraps two commands back-to-back (printing their logs live while teeing to files):

1. `python -m evaluations.evaluations.benchmark run repliqa ...` (baseline HaikuRAG
   retrieval + QA grading) – logs land in `logs/evaluations/<timestamp>_baseline.log`.
2. `python scripts/compare_pipelines.py ...` (neighbor-aware telemetry) – logs plus the JSON metrics land in the same folder.

Override `--questions-file` with whichever list matches the dataset slice you want,
and point `--config/--db` at other pipelines if needed.

## 4. Expand to answer-level scoring

Once retrieval quality stabilizes, feed the questions and retrieved contexts
into the quiz runner (`tests/unit/test_chunk_selection.py` exercises the
selector) or the `haiku.rag/evaluations` harness. The JSON emitted by
`compare_pipelines.py` contains enough provenance (document IDs, URIs, scores)
to automate full-answer judge calls and cost/time accounting per model.

Recommended next steps:

1. Build a `logs/run-<timestamp>/` folder that stores:
   - the raw comparison output
   - the final judged answers (via `soliplex.quizzes`)
   - the git SHAs for both repos
2. Promote representative runs into a regression suite so CI can detect
   retrieval or scoring drift.

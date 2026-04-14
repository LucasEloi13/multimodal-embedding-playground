# Copilot Instructions for `adidas-image-match`

## Build, test, and lint commands

This repository is pipeline-first and currently does **not** define a dedicated lint or unit-test suite (`pytest`, `ruff`, `mypy`, etc. are not configured in `pyproject.toml` and there is no `tests/` directory).

Use these runnable project commands:

```bash
# Install dependencies
uv sync

# Build text FAISS index
uv run python src/pipeline/run_reference_indexing.py --embed-text

# Build image FAISS index
uv run python src/pipeline/run_reference_indexing.py --embed-images

# Run retrieval evaluation (config-driven)
uv run python src/pipeline/run_retrieval_evaluation.py

# Compare rerank strategies on the same sample
uv run python src/pipeline/run_retrieval_strategy_comparison.py

# Run LLM evaluation over query/top1 pairs
uv run python src/pipeline/run_llm_evaluation.py

# Convert one usage log file (single-file run)
uv run python src/pipeline/run_usage_log_structuring.py \
  --input-file src/infra/llm/usage_logs/usage_YYYYMMDD_HHMMSS.jsonl
```

If you need a minimal smoke run while editing pipeline code, use:

```bash
uv run python src/pipeline/run_reference_indexing.py --embed-text --limit 10
```

## High-level architecture

The repository implements a configurable product-matching workflow with three main stages:

1. **Reference indexing** (`src/pipeline/run_reference_indexing.py`)
   - Loads catalog products from `data/items.jsonl` using `ReferenceProductLoader`.
   - Builds text embeddings (`TextEmbeddingService` + `TextEmbeddingProcessor`) and/or image embeddings (`ImageEmbeddingService`).
   - Persists FAISS index files in `data/my_index/` with sidecar metadata JSONL.

2. **Retrieval + reranking evaluation** (`src/pipeline/run_retrieval_evaluation.py`)
   - Reads runtime settings from `src/infra/config/production.yml` (`test` section) and model/provider settings from `src/infra/config/embedding_models.yml`.
   - Samples query listings from `data/scrapped_listings` (or uses `--query-file`).
   - Retrieves by text against the text FAISS index, then reranks with one of `mean`, `text_only`, `image_only` using image cosine similarity when needed.
   - Writes structured output to `data/debug/retrieval_evaluation.*.json` and auto-splits large outputs into `part_###` files.

3. **LLM evaluation + usage accounting** (`src/pipeline/run_llm_evaluation.py`, `src/embedding/services/llm_evaluation.py`)
   - Evaluates query/candidate pairs through provider modules (currently Gemini under `src/infra/llm/providers/gemini.py`).
   - Logs per-call token usage to JSONL and also emits structured usage summaries (`usage_*.json` and `usage_totals.json`).

## Key codebase conventions

- **Config-first behavior**: defaults come from YAML config files; CLI flags are overrides. Preserve that precedence when adding options.
- **Dual import-path fallback**: many modules try `core.*` / `infra.*` imports, then fallback to `src.core.*` / `src.infra.*`. Keep this pattern in pipeline/service entrypoints for PYTHONPATH flexibility.
- **Index naming contract**: index filenames are derived from provider/model via sanitized model names (for example `reference_text_openai-text-embedding-3-large.faiss`). Retrieval scripts rely on this convention to auto-resolve index paths.
- **Metadata sidecar contract**: FAISS indexes are written with `.faiss.meta.jsonl` sidecars; downstream retrieval reads metadata for SKU lookup and candidate enrichment.
- **Resilient batch processing**: image/text embedding flows favor partial progress (batch fallback to sequential, per-item warnings) instead of failing the whole run on one bad record.
- **Text normalization rules matter for retrieval quality**: `TextNormalizer` lowercases, removes diacritics/non-alphanumerics, caps description length, and builds a Portuguese-flavored sentence (`marca`, `cor`, `mpn`) before text embedding.

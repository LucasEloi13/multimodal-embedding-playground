# adidas-image-match

## Pipelines

### 1) Build reference indexes

Generate text index:

```bash
uv run python src/pipeline/run_reference_indexing.py --embed-text
```

Generate image index (provider from `src/infra/config/embedding_models.yml`):

```bash
uv run python src/pipeline/run_reference_indexing.py --embed-images
```

### 2) Run retrieval evaluation

Main pipeline (config-driven):

```bash
uv run python src/pipeline/run_retrieval_evaluation.py
```

This pipeline will:

1. Read runtime settings from `src/infra/config/production.yml` (`test` section).
2. Collect all JSON files from `data/scrapped_listings`.
3. Generate a random query sample based on `test.sample_size`.
4. Resolve default text/image FAISS indexes from `src/infra/config/embedding_models.yml`.
5. Run retrieval for `test.top_k`.
6. Rerank using `test.rerank_strategy` and weights.
7. Rerank and keep only top `test.rerank_k` candidates per listing.
8. Save output JSON to `data/debug/retrieval_evaluation.{datetime}.json`.
9. Split output automatically when results exceed `test.max_samples_per_output_file`.

## Runtime config

`src/infra/config/production.yml`:

```yml
pipeline:
	text_embedding_batch: 250
	image_embedding_batch: 64
	image_download_workers: 32
	image_process_workers: 8

test:
	sample_size: 1000
	top_k: 15
	rerank_k: 5
	rerank_strategy: mean
	rerank_image_weight: 0.5
	rerank_text_weight: 0.5
	workers: 4
	max_samples_per_output_file: 100
```

Each returned candidate contains:

1. `text_rank`
2. `image_rank`
3. `overall_rank`

Supported rerank strategies:

1. `mean`: weighted mean of text and image scores.
2. `text_only`: rerank only by text score.
3. `image_only`: rerank only by image cosine score.

## Optional overrides

You can override defaults when needed:

```bash
uv run python src/pipeline/run_retrieval_evaluation.py \
	--sample-size 200 \
	--top-k 20 \
	--rerank-k 10 \
	--rerank-strategy mean \
	--rerank-image-weight 0.7 \
	--rerank-text-weight 0.3 \
	--workers 8 \
	--max-samples-per-output-file 100 \
	--image-index-path data/my_index/reference_image_fashion-clip-patrickjohncyh-fashion-clip.faiss
```

If an expected index file is missing, the pipeline will fail with an actionable message showing the command to generate it or how to pass an explicit index path.

## Usage logs to structured JSON

Convert usage logs from JSONL to formatted JSON with summary metrics:

```bash
uv run python src/pipeline/run_usage_log_structuring.py
```

To convert only one file:

```bash
uv run python src/pipeline/run_usage_log_structuring.py \
	--input-file src/infra/llm/usage_logs/usage_20260414_134214.jsonl
```

"""Run retrieval evaluation for scrapped listings using text+image reranking."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import logging
import os
from queue import Queue
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from core.utils.reference_product_loader import JsonReferenceProductParser
    from core.utils.scrapped_listing_sampler import create_random_sample_file
    from core.utils.text_normalizer import TextNormalizer
    from embedding.services.image_embedder import ImageEmbeddingService
    from embedding.services.text_embedder import TextEmbeddingService
    from infra.vector_db.faiss import FaissVectorStore
except ModuleNotFoundError:
    from src.core.utils.reference_product_loader import JsonReferenceProductParser
    from src.core.utils.scrapped_listing_sampler import create_random_sample_file
    from src.core.utils.text_normalizer import TextNormalizer
    from src.embedding.services.image_embedder import ImageEmbeddingService
    from src.embedding.services.text_embedder import TextEmbeddingService
    from src.infra.vector_db.faiss import FaissVectorStore


logger = logging.getLogger(__name__)


class ImageEmbeddingServicePool:
    def __init__(
        self,
        size: int,
        config_path: Path,
        seed_service: ImageEmbeddingService | None = None,
    ) -> None:
        pool_size = max(1, int(size))
        self._queue: Queue[ImageEmbeddingService] = Queue(maxsize=pool_size)

        if seed_service is not None:
            self._queue.put(seed_service)
            remaining = pool_size - 1
        else:
            remaining = pool_size

        for _ in range(remaining):
            self._queue.put(ImageEmbeddingService(config_path=config_path))

    @contextmanager
    def checkout(self):
        service = self._queue.get()
        try:
            yield service
        finally:
            self._queue.put(service)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scrapped listing retrieval")
    parser.add_argument(
        "--production-config",
        type=Path,
        default=PROJECT_ROOT / "src" / "infra" / "config" / "production.yml",
        help="Runtime config file",
    )
    parser.add_argument(
        "--embedding-config",
        type=Path,
        default=PROJECT_ROOT / "src" / "infra" / "config" / "embedding_models.yml",
        help="Embedding model config file",
    )
    parser.add_argument(
        "--scrapped-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "scrapped_listings",
        help="Directory with scrapped listing json files",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Override sample size")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k retrieval")
    parser.add_argument("--rerank-k", type=int, default=None, help="Override top-k after rerank")
    parser.add_argument(
        "--rerank-strategy",
        type=str,
        choices=["mean", "text_only", "image_only"],
        default=None,
        help="Override rerank strategy",
    )
    parser.add_argument(
        "--rerank-image-weight",
        type=float,
        default=None,
        help="Override image weight for mean rerank strategy",
    )
    parser.add_argument(
        "--rerank-text-weight",
        type=float,
        default=None,
        help="Override text weight for mean rerank strategy",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="Optional existing query file. If omitted, sample is generated automatically.",
    )
    parser.add_argument(
        "--sample-output-file",
        type=Path,
        default=None,
        help="Optional path for generated sample query file",
    )
    parser.add_argument(
        "--text-index-path",
        type=Path,
        default=None,
        help="Optional explicit text index path",
    )
    parser.add_argument(
        "--image-index-path",
        type=Path,
        default=None,
        help="Optional explicit image index path",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--reference-catalog-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "items.jsonl",
        help="Reference catalog used to enrich candidate fields by SKU",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of query workers",
    )
    parser.add_argument(
        "--max-samples-per-output-file",
        type=int,
        default=None,
        help="Override max number of results per output JSON file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _sanitize_model_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")


def _resolve_text_index_path(
    explicit_path: Path | None,
    embedding_config: dict[str, Any],
) -> Path:
    if explicit_path is not None:
        return explicit_path

    text_cfg = embedding_config.get("text_embedding", {})
    provider = str(text_cfg.get("provider", "openai")).lower()
    model = str(text_cfg.get("model", "text-embedding-3-large"))
    model_name = f"{provider}-{model}"

    filename = f"reference_text_{_sanitize_model_name(model_name)}.faiss"
    return PROJECT_ROOT / "data" / "my_index" / filename


def _resolve_image_index_path(
    explicit_path: Path | None,
    image_service: ImageEmbeddingService,
) -> Path:
    if explicit_path is not None:
        return explicit_path

    filename = f"reference_image_{_sanitize_model_name(image_service.model_name)}.faiss"
    return PROJECT_ROOT / "data" / "my_index" / filename


def _validate_index_exists(
    index_path: Path,
    embedding_type: str,
    override_flag: str,
) -> None:
    if index_path.exists():
        return

    command = (
        "uv run python src/pipeline/run_reference_indexing.py --embed-images"
        if embedding_type == "image"
        else "uv run python src/pipeline/run_reference_indexing.py --embed-text"
    )
    raise FileNotFoundError(
        f"{embedding_type.capitalize()} index not found: {index_path}. "
        f"Generate it with '{command}' or pass '{override_flag}' with a specific index file."
    )


def _load_queries(query_file: Path) -> list[dict[str, Any]]:
    parser = JsonReferenceProductParser()
    payload = query_file.read_text(encoding="utf-8")
    return parser.parse(payload)


def _build_query_text(listing_payload: dict[str, Any], normalizer: TextNormalizer) -> str:
    normalized = normalizer.normalize_product_text(listing_payload)
    return normalizer.build_sentence(normalized)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _find_reference_image_vector_by_sku(image_store: FaissVectorStore, sku: str) -> np.ndarray | None:
    indices = image_store.find_indices_by_metadata("sku", sku)
    if not indices:
        return None
    return image_store.get_vector(indices[0])


def _compute_rerank_score(
    text_score: float,
    image_score: float | None,
    strategy: str,
    text_weight: float,
    image_weight: float,
) -> float:
    if strategy == "text_only":
        return text_score

    if strategy == "image_only":
        return float(image_score) if image_score is not None else float("-inf")

    if image_score is None:
        return text_score

    total_weight = text_weight + image_weight
    if total_weight <= 0:
        return (text_score + float(image_score)) / 2.0

    return (text_score * text_weight + float(image_score) * image_weight) / total_weight


def _configure_external_noise() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message=".*You are sending unauthenticated requests to the HF Hub.*",
    )

    try:
        transformers_logging = __import__("transformers.utils.logging", fromlist=["set_verbosity_error"])
        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "data" / "debug" / f"retrieval_evaluation.{timestamp}.json"


def _assign_image_ranks(candidates: list[dict[str, Any]]) -> None:
    ordered = sorted(
        candidates,
        key=lambda item: item["image_cosine_score"] if item["image_cosine_score"] is not None else float("-inf"),
        reverse=True,
    )
    for rank, item in enumerate(ordered, start=1):
        item["image_rank"] = rank


def _write_chunked_output(
    output_file: Path,
    results: list[dict[str, Any]],
    max_samples_per_file: int,
) -> list[Path]:
    chunk_size = max(1, int(max_samples_per_file))
    chunks = [results[i : i + chunk_size] for i in range(0, len(results), chunk_size)]
    if not chunks:
        chunks = [[]]

    written: list[Path] = []
    if len(chunks) == 1:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(chunks[0], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(output_file)
        return written

    suffix = output_file.suffix or ".json"
    stem = output_file.stem if output_file.suffix else output_file.name
    total_parts = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        part_path = output_file.with_name(f"{stem}.part_{idx:03d}{suffix}")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_text(json.dumps(chunk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(part_path)

    return written


def _run_query(
    query: dict[str, Any],
    top_k: int,
    rerank_k: int,
    rerank_strategy: str,
    rerank_text_weight: float,
    rerank_image_weight: float,
    text_service: TextEmbeddingService,
    image_service: ImageEmbeddingService,
    image_service_pool: ImageEmbeddingServicePool | None,
    text_store: FaissVectorStore,
    image_store: FaissVectorStore,
    normalizer: TextNormalizer,
) -> dict[str, Any]:
    query_text = _build_query_text(query, normalizer)
    query_text_embedding = text_service.embed_text(query_text)
    text_hits = text_store.search(query_text_embedding, top_k=top_k)

    needs_image_signal = rerank_strategy in {"mean", "image_only"}
    query_image_url = ((query.get("mainImage") or {}).get("url"))
    query_image_embedding = None
    if needs_image_signal and query_image_url:
        if image_service_pool is None:
            query_image_embedding = image_service.embed_image_url(query_image_url)
        else:
            with image_service_pool.checkout() as pooled_service:
                query_image_embedding = pooled_service.embed_image_url(query_image_url)

    candidates: list[dict[str, Any]] = []
    for rank, hit in enumerate(text_hits, start=1):
        metadata = hit.get("metadata") or {}
        sku = metadata.get("sku")
        if not isinstance(sku, str):
            continue

        text_score = float(hit.get("score", 0.0))
        image_score: float | None = None

        if query_image_embedding is not None:
            reference_vector = _find_reference_image_vector_by_sku(image_store=image_store, sku=sku)
            if reference_vector is not None:
                image_score = _cosine_similarity(query_image_embedding, reference_vector)

        rerank_score = _compute_rerank_score(
            text_score=text_score,
            image_score=image_score,
            strategy=rerank_strategy,
            text_weight=rerank_text_weight,
            image_weight=rerank_image_weight,
        )

        candidates.append(
            {
                "text_rank": rank,
                "sku": sku,
                "url": metadata.get("url"),
                "canonical_url": metadata.get("canonical_url"),
                "text_score": text_score,
                "image_cosine_score": image_score,
                "rerank_score": rerank_score,
            }
        )

    _assign_image_ranks(candidates)
    reranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:rerank_k]
    for idx, item in enumerate(reranked, start=1):
        item["overall_rank"] = idx

    return {
        "listing": {
            "sku": query.get("sku"),
            "name": query.get("name"),
            "url": query.get("url"),
            "canonical_url": query.get("canonicalUrl"),
            "brand": (
                (query.get("brand") or {}).get("name")
                if isinstance(query.get("brand"), dict)
                else query.get("brand")
            ),
            "description": query.get("description"),
            "additional_properties": query.get("additionalProperties")
            or query.get("additional_properties")
            or [],
            "main_image_url": query_image_url,
        },
        "candidates": reranked,
    }


def _run_query_safe(
    query: dict[str, Any],
    top_k: int,
    rerank_k: int,
    rerank_strategy: str,
    rerank_text_weight: float,
    rerank_image_weight: float,
    text_service: TextEmbeddingService,
    image_service: ImageEmbeddingService,
    image_service_pool: ImageEmbeddingServicePool | None,
    text_store: FaissVectorStore,
    image_store: FaissVectorStore,
    normalizer: TextNormalizer,
) -> dict[str, Any] | None:
    try:
        return _run_query(
            query=query,
            top_k=top_k,
            rerank_k=rerank_k,
            rerank_strategy=rerank_strategy,
            rerank_text_weight=rerank_text_weight,
            rerank_image_weight=rerank_image_weight,
            text_service=text_service,
            image_service=image_service,
            image_service_pool=image_service_pool,
            text_store=text_store,
            image_store=image_store,
            normalizer=normalizer,
        )
    except Exception as exc:  # pragma: no cover - runtime resilience
        logger.warning(
            "Skipping query due to processing error (sku=%s, image_url=%s): %s",
            query.get("sku"),
            ((query.get("mainImage") or {}).get("url")),
            exc,
        )
        return None


def _load_reference_catalog_by_sku(catalog_file: Path) -> dict[str, dict[str, Any]]:
    if not catalog_file.exists():
        logger.warning("Reference catalog not found: %s", catalog_file)
        return {}

    parser = JsonReferenceProductParser()
    try:
        records = parser.parse(catalog_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse reference catalog (%s): %s", catalog_file, exc)
        return {}

    catalog: dict[str, dict[str, Any]] = {}
    for record in records:
        sku = record.get("sku")
        if not isinstance(sku, str) or not sku:
            continue

        raw_brand = record.get("brand")
        if isinstance(raw_brand, dict):
            brand_value = raw_brand.get("name")
        elif isinstance(raw_brand, str):
            brand_value = raw_brand
        else:
            brand_value = None

        image_url = None
        main_image = record.get("mainImage")
        if isinstance(main_image, dict):
            image_url = main_image.get("url")
        if not image_url:
            images = record.get("images")
            if isinstance(images, list) and images and isinstance(images[0], dict):
                image_url = images[0].get("url")

        catalog[sku] = {
            "name": record.get("name"),
            "brand": brand_value,
            "description": record.get("description"),
            "image_url": image_url,
        }

    return catalog


def _format_output_records(
    raw_results: list[dict[str, Any]],
    reference_catalog: dict[str, dict[str, Any]],
    created_at: str,
) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []

    for result in raw_results:
        listing = result.get("listing") or {}
        query_payload = {
            "sku": listing.get("sku"),
            "name": listing.get("name"),
            "brand": listing.get("brand"),
            "description": listing.get("description"),
            "additional_properties": listing.get("additional_properties") or [],
            "image_url": listing.get("main_image_url"),
        }

        formatted_candidates: list[dict[str, Any]] = []
        for candidate in result.get("candidates") or []:
            candidate_sku = candidate.get("sku")
            catalog_info = reference_catalog.get(candidate_sku, {}) if isinstance(candidate_sku, str) else {}

            formatted_candidates.append(
                {
                    "candidate_sku": candidate_sku,
                    "name": catalog_info.get("name"),
                    "brand": catalog_info.get("brand"),
                    "description": catalog_info.get("description"),
                    "image_url": catalog_info.get("image_url"),
                    "features": {
                        "text_similarity": float(candidate.get("text_score") or 0.0),
                        "image_similarity": float(candidate.get("image_cosine_score") or 0.0),
                    },
                    "label": {
                        "is_match": None,
                        "source": None,
                        "review_notes": None,
                    },
                }
            )

        formatted.append(
            {
                "query": query_payload,
                "candidates": formatted_candidates,
                "metadata": {
                    "created_at": created_at,
                },
            }
        )

    return formatted


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("[1/7] Loading configs")
    production = _load_yaml(args.production_config)
    embedding_cfg = _load_yaml(args.embedding_config)
    test_cfg = production.get("test", {}) if isinstance(production.get("test", {}), dict) else {}

    sample_size = int(args.sample_size or test_cfg.get("sample_size", 100))
    top_k = int(args.top_k or test_cfg.get("top_k", 15))
    rerank_k = int(args.rerank_k or test_cfg.get("rerank_k", 5))
    rerank_strategy = str(args.rerank_strategy or test_cfg.get("rerank_strategy", "mean"))
    rerank_image_weight = float(args.rerank_image_weight or test_cfg.get("rerank_image_weight", 0.5))
    rerank_text_weight = float(args.rerank_text_weight or test_cfg.get("rerank_text_weight", 0.5))
    workers = int(args.workers or test_cfg.get("workers", 1))
    max_samples_per_output_file = int(
        args.max_samples_per_output_file or test_cfg.get("max_samples_per_output_file", 100)
    )

    logger.info("[2/7] Configuring model logging/noise")
    _configure_external_noise()

    logger.info("[3/7] Initializing embedding services")
    text_service = TextEmbeddingService(config_path=args.embedding_config)
    image_service = ImageEmbeddingService(config_path=args.embedding_config)

    logger.info("[4/7] Resolving index files")
    text_index_path = _resolve_text_index_path(args.text_index_path, embedding_config=embedding_cfg)
    image_index_path = _resolve_image_index_path(args.image_index_path, image_service=image_service)
    _validate_index_exists(text_index_path, embedding_type="text", override_flag="--text-index-path")
    _validate_index_exists(image_index_path, embedding_type="image", override_flag="--image-index-path")

    text_store = FaissVectorStore.from_files(text_index_path)
    image_store = FaissVectorStore.from_files(image_index_path)

    logger.info("[5/7] Preparing query sample")
    query_file: Path
    if args.query_file is not None:
        query_file = args.query_file
    else:
        query_file, sample = create_random_sample_file(
            scrapped_dir=args.scrapped_dir,
            sample_size=sample_size,
            output_file=args.sample_output_file,
        )
        logger.info("Sample generated: file=%s size=%s", query_file, len(sample))

    queries = _load_queries(query_file)
    if not queries:
        raise RuntimeError(f"No queries found in {query_file}")

    logger.info("[6/7] Running retrieval for %s queries", len(queries))
    normalizer = TextNormalizer()
    results: list[dict[str, Any]] = []
    image_service_pool: ImageEmbeddingServicePool | None = None
    if workers > 1:
        image_service_pool = ImageEmbeddingServicePool(
            size=workers,
            config_path=args.embedding_config,
            seed_service=image_service,
        )

    total = len(queries)
    if workers <= 1:
        for idx, query in enumerate(queries, start=1):
            result = _run_query_safe(
                query=query,
                top_k=top_k,
                rerank_k=rerank_k,
                rerank_strategy=rerank_strategy,
                rerank_text_weight=rerank_text_weight,
                rerank_image_weight=rerank_image_weight,
                text_service=text_service,
                image_service=image_service,
                image_service_pool=image_service_pool,
                text_store=text_store,
                image_store=image_store,
                normalizer=normalizer,
            )
            if result is not None:
                results.append(result)
            if idx == total or idx % 25 == 0:
                logger.info("Progress: %s/%s queries processed", idx, total)
    else:
        ordered_results: list[dict[str, Any] | None] = [None] * total
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    _run_query_safe,
                    query,
                    top_k,
                    rerank_k,
                    rerank_strategy,
                    rerank_text_weight,
                    rerank_image_weight,
                    text_service,
                    image_service,
                    image_service_pool,
                    text_store,
                    image_store,
                    normalizer,
                ): idx
                for idx, query in enumerate(queries)
            }

            processed = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                ordered_results[idx] = future.result()
                processed += 1
                if processed == total or processed % 25 == 0:
                    logger.info("Progress: %s/%s queries processed", processed, total)

        results = [item for item in ordered_results if item is not None]

    logger.info("[7/7] Writing output")
    output_file = args.output_file or _default_output_path()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reference_catalog = _load_reference_catalog_by_sku(args.reference_catalog_file)
    formatted_results = _format_output_records(
        raw_results=results,
        reference_catalog=reference_catalog,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    written_files = _write_chunked_output(
        output_file=output_file,
        results=formatted_results,
        max_samples_per_file=max_samples_per_output_file,
    )
    logger.info("Done: %s file(s) written", len(written_files))
    for path in written_files:
        logger.info("Output: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

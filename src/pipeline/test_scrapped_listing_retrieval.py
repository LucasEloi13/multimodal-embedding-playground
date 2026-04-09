"""Test retrieval for scrapped listings against reference text index."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from core.utils.reference_product_loader import JsonReferenceProductParser
    from core.utils.text_normalizer import TextNormalizer
    from embedding.services.image_embedder import ImageEmbeddingService
    from embedding.services.text_embedder import TextEmbeddingService
    from infra.vector_db.faiss import FaissVectorStore
except ModuleNotFoundError:
    from src.core.utils.reference_product_loader import JsonReferenceProductParser
    from src.core.utils.text_normalizer import TextNormalizer
    from src.embedding.services.image_embedder import ImageEmbeddingService
    from src.embedding.services.text_embedder import TextEmbeddingService
    from src.infra.vector_db.faiss import FaissVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve closest reference products for scrapped listing")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "scrapped_listings" / "example.json",
        help="JSON/JSONL file with listing(s) to query",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "src" / "infra" / "config" / "embedding_models.yml",
        help="Embedding provider config",
    )
    parser.add_argument(
        "--text-index-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "my_index" / "reference_text_openai-text-embedding-3-large.faiss",
        help="Path to text FAISS index",
    )
    parser.add_argument(
        "--image-index-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "my_index" / "reference_image_dino-vit_base_patch16_dinov3.faiss",
        help="Path to image FAISS index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest products",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to save JSON output. If omitted, creates data/debug/retrieval_output.{datetime}.json",
    )
    return parser.parse_args()


def load_queries(query_file: Path) -> list[dict[str, Any]]:
    parser = JsonReferenceProductParser()
    raw = query_file.read_text(encoding="utf-8")
    return parser.parse(raw)


def build_query_text(listing_payload: dict[str, Any], normalizer: TextNormalizer) -> str:
    normalized = normalizer.normalize_product_text(listing_payload)
    return normalizer.build_sentence(normalized)


def run_text_retrieval(
    listing_payload: dict[str, Any],
    text_service: TextEmbeddingService,
    vector_store: FaissVectorStore,
    top_k: int,
    normalizer: TextNormalizer,
) -> list[dict[str, Any]]:
    query_text = build_query_text(listing_payload, normalizer=normalizer)
    query_embedding = text_service.embed_text(query_text)
    return vector_store.search(query_embedding, top_k=top_k)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def find_reference_image_vector_by_sku(
    image_store: FaissVectorStore,
    sku: str,
) -> np.ndarray | None:
    indices = image_store.find_indices_by_metadata("sku", sku)
    if not indices:
        return None
    return image_store.get_vector(indices[0])


def run_image_similarity_for_text_hits(
    listing_payload: dict[str, Any],
    text_hits: list[dict[str, Any]],
    image_service: ImageEmbeddingService,
    image_store: FaissVectorStore,
) -> list[dict[str, Any]]:
    main_image = listing_payload.get("mainImage") or {}
    query_image_url = main_image.get("url")
    if not query_image_url:
        return []

    query_vec = image_service.embed_image_url(query_image_url)
    results: list[dict[str, Any]] = []

    for rank, hit in enumerate(text_hits, start=1):
        metadata = hit.get("metadata") or {}
        sku = metadata.get("sku")
        if not isinstance(sku, str):
            continue

        ref_vec = find_reference_image_vector_by_sku(image_store=image_store, sku=sku)
        if ref_vec is None:
            results.append({"rank": rank, "sku": sku, "cosine": None})
            continue

        score = cosine_similarity(query_vec, ref_vec)
        results.append({"rank": rank, "sku": sku, "cosine": score})

    return results


def build_debug_output(
    query: dict[str, Any],
    text_hits: list[dict[str, Any]],
    image_scores: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    image_by_sku = {
        item["sku"]: item.get("cosine")
        for item in image_scores
        if isinstance(item.get("sku"), str)
    }

    candidates: list[dict[str, Any]] = []
    for rank, hit in enumerate(text_hits, start=1):
        metadata = hit.get("metadata") or {}
        sku = metadata.get("sku")
        if not isinstance(sku, str):
            continue

        candidates.append(
            {
                "rank": rank,
                "sku": sku,
                "url": metadata.get("url"),
                "canonical_url": metadata.get("canonical_url"),
                "text_score": float(hit.get("score", 0.0)),
                "image_cosine_score": image_by_sku.get(sku),
            }
        )

    return {
        "listing": {
            "sku": query.get("sku"),
            "name": query.get("name"),
            "url": query.get("url"),
            "canonical_url": query.get("canonicalUrl"),
            "main_image_url": (query.get("mainImage") or {}).get("url"),
        },
        "top_k": top_k,
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()

    output_file = args.output_file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROJECT_ROOT / "data" / "debug" / f"retrieval_output.{timestamp}.json"

    queries = load_queries(args.query_file)
    if not queries:
        print("No query listings found")
        return 1

    text_service = TextEmbeddingService(config_path=args.config)
    image_service = ImageEmbeddingService(config_path=args.config)
    text_store = FaissVectorStore.from_files(args.text_index_path)
    image_store = FaissVectorStore.from_files(args.image_index_path)
    normalizer = TextNormalizer()

    outputs: list[dict[str, Any]] = []

    for query in queries:
        text_hits = run_text_retrieval(
            listing_payload=query,
            text_service=text_service,
            vector_store=text_store,
            top_k=args.top_k,
            normalizer=normalizer,
        )

        image_scores = run_image_similarity_for_text_hits(
            listing_payload=query,
            text_hits=text_hits,
            image_service=image_service,
            image_store=image_store,
        )

        outputs.append(
            build_debug_output(
                query=query,
                text_hits=text_hits,
                image_scores=image_scores,
                top_k=args.top_k,
            )
        )

    payload: dict[str, Any]
    if len(outputs) == 1:
        payload = outputs[0]
    else:
        payload = {"results": outputs}

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved output file: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

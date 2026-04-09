"""Batch processor for text embeddings using normalized product sentences."""

from __future__ import annotations

import logging
from typing import Any

try:
    from core.utils.text_normalizer import TextNormalizer
except ModuleNotFoundError:
    from src.core.utils.text_normalizer import TextNormalizer


logger = logging.getLogger(__name__)


class TextEmbeddingProcessor:
    def __init__(self, text_embedder: Any, text_normalizer: TextNormalizer | None = None) -> None:
        self.text_embedder = text_embedder
        self.text_normalizer = text_normalizer or TextNormalizer()

    def process(
        self,
        products: list[Any],
        batch_size: int,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        stats: dict[str, Any] = {"successful": 0, "errors": []}
        product_data: list[dict[str, Any]] = []

        for product in products:
            try:
                normalized = self.text_normalizer.normalize_reference_product(product)
                sentence = self.text_normalizer.build_sentence(normalized)
                product_data.append(
                    {
                        "product": product,
                        "text": sentence,
                        "product_text": normalized,
                    }
                )
            except Exception as exc:
                stats["errors"].append(
                    {"sku": getattr(product, "sku", None), "error": f"Error preparing text: {exc}"}
                )
                logger.warning("Error preparing text %s: %s", getattr(product, "sku", None), exc)

        embeddings: list[Any] = []
        metadatas: list[dict[str, Any]] = []

        use_batch = hasattr(self.text_embedder, "embed_texts_batch")

        if use_batch:
            logger.info("Using batch text processing (batch_size=%s)", batch_size)
            num_batches = (len(product_data) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(product_data))
                batch = product_data[start:end]

                try:
                    texts = [item["text"] for item in batch]
                    batch_embeddings = self.text_embedder.embed_texts_batch(texts, batch_size=batch_size)

                    for item, embedding in zip(batch, batch_embeddings, strict=False):
                        product = item["product"]
                        product_text = item["product_text"]
                        embeddings.append(embedding)
                        metadatas.append(self._build_metadata(product, product_text))
                        stats["successful"] += 1
                except Exception as exc:
                    logger.warning(
                        "Error in text batch %s, falling back to sequential: %s",
                        batch_idx,
                        exc,
                    )
                    self._process_batch_sequential(batch, embeddings, metadatas, stats)

                if progress_callback:
                    progress_callback(end, len(product_data))
        else:
            logger.info("Text embedder without batch support, processing sequentially")
            self._process_batch_sequential(product_data, embeddings, metadatas, stats)

        stats["embeddings"] = embeddings
        stats["metadatas"] = metadatas
        return stats

    def _process_batch_sequential(
        self,
        batch: list[dict[str, Any]],
        embeddings: list[Any],
        metadatas: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> None:
        for item in batch:
            product = item["product"]
            try:
                embedding = self.text_embedder.embed_text(item["text"])
                embeddings.append(embedding)
                metadatas.append(self._build_metadata(product, item["product_text"]))
                stats["successful"] += 1
            except Exception as exc:
                stats["errors"].append(
                    {
                        "sku": getattr(product, "sku", None),
                        "error": str(exc),
                    }
                )
                logger.warning("Error processing text %s: %s", getattr(product, "sku", None), exc)

    def _build_metadata(self, product: Any, product_text: dict[str, Any]) -> dict[str, Any]:
        brand_name = ""
        if getattr(product, "brand", None) is not None:
            brand_name = getattr(product.brand, "name", "") or ""

        image_url = None
        if getattr(product, "main_image", None) is not None:
            image_url = getattr(product.main_image, "url", None)

        description = getattr(product, "description", "") or ""

        return {
            "sku": getattr(product, "sku", None),
            "name": getattr(product, "name", None),
            "brand": brand_name,
            "description": description[:500],
            "url": getattr(product, "url", None),
            "image_url": image_url,
            "price": getattr(product, "price", None),
            "product_text": product_text,
            "type": "reference",
            "embedding_type": "text",
        }

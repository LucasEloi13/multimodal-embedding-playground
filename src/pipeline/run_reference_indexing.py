"""Pipeline to build reference product vector indexes."""

from __future__ import annotations

import argparse
import importlib
import logging
import re
import sys
from pathlib import Path
from typing import Any

try:
	import yaml
except ImportError as exc:  # pragma: no cover - dependency check
	raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))


logger = logging.getLogger(__name__)


def _resolve_symbols() -> tuple[Any, Any, Any, Any, Any, Any]:
	candidates = [
		(
			"core.entities.reference_product",
			"core.utils.reference_product_loader",
			"embedding.services.image_embedder",
			"embedding.services.text_embedder",
			"embedding.services.text_embedding_processor",
			"infra.vector_db.faiss",
		),
		(
			"src.core.entities.reference_product",
			"src.core.utils.reference_product_loader",
			"src.embedding.services.image_embedder",
			"src.embedding.services.text_embedder",
			"src.embedding.services.text_embedding_processor",
			"src.infra.vector_db.faiss",
		),
	]

	for entity_mod, loader_mod, image_mod, text_mod, text_processor_mod, vector_mod in candidates:
		try:
			reference_product = importlib.import_module(entity_mod).ReferenceProduct
			reference_loader = importlib.import_module(loader_mod).ReferenceProductLoader
			image_service = importlib.import_module(image_mod).ImageEmbeddingService
			text_service = importlib.import_module(text_mod).TextEmbeddingService
			text_processor = importlib.import_module(text_processor_mod).TextEmbeddingProcessor
			vector_store = importlib.import_module(vector_mod).FaissVectorStore
			return (
				reference_product,
				reference_loader,
				image_service,
				text_service,
				text_processor,
				vector_store,
			)
		except ModuleNotFoundError:
			continue

	raise ModuleNotFoundError("Could not resolve project modules for pipeline imports")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Reference product indexing pipeline")
	parser.add_argument(
		"--input-file",
		type=Path,
		default=PROJECT_ROOT / "data" / "items.jsonl",
		help="Input file with reference products",
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=PROJECT_ROOT / "src" / "infra" / "config" / "embedding_models.yml",
		help="Embedding provider configuration file",
	)
	parser.add_argument(
		"--production-config",
		type=Path,
		default=PROJECT_ROOT / "src" / "infra" / "config" / "production.yml",
		help="Pipeline runtime configuration file",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=PROJECT_ROOT / "data" / "my_index",
		help="Directory where index files are saved",
	)
	parser.add_argument(
		"--embed-text",
		action="store_true",
		help="Create text embedding index",
	)
	parser.add_argument(
		"--embed-images",
		action="store_true",
		help="Create image embedding index",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Limit number of products for quick tests",
	)
	parser.add_argument(
		"--image-timeout",
		type=float,
		default=20.0,
		help="HTTP timeout for downloading product images",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="Logging level",
	)
	parser.add_argument(
		"--text-batch-size",
		type=int,
		default=None,
		help="Batch size for text embedding generation",
	)
	parser.add_argument(
		"--image-batch-size",
		type=int,
		default=None,
		help="Batch size for image embedding generation",
	)
	parser.add_argument(
		"--image-download-workers",
		type=int,
		default=None,
		help="Concurrent workers for image download",
	)
	parser.add_argument(
		"--image-process-workers",
		type=int,
		default=None,
		help="Workers used on image embedding fallback processing",
	)
	return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	payload = yaml.safe_load(path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		return {}
	return payload


def sanitize_model_name(value: str) -> str:
	return re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")


def pick_main_image_url(product: Any) -> str | None:
	if product.main_image and product.main_image.url:
		return product.main_image.url
	if product.images:
		first = product.images[0]
		if first and first.url:
			return first.url
	return None


def build_text_metadata(product: Any) -> dict[str, Any]:
	return {
		"type": "text",
		"sku": product.sku,
		"name": product.name,
		"url": product.url,
		"canonical_url": product.canonical_url,
	}


def build_image_metadata(product: Any, image_url: str) -> dict[str, Any]:
	return {
		"type": "image",
		"sku": product.sku,
		"name": product.name,
		"url": product.url,
		"canonical_url": product.canonical_url,
		"image_url": image_url,
	}


def create_text_index(
	products: list[Any],
	config_path: Path,
	output_dir: Path,
	text_service_cls: Any,
	text_processor_cls: Any,
	vector_store_cls: Any,
	batch_size: int,
) -> Path:
	service = text_service_cls(config_path=config_path)
	processor = text_processor_cls(text_embedder=service)
	result = processor.process(products=products, batch_size=batch_size)
	vectors = result.get("embeddings", [])
	metadata = result.get("metadatas", [])
	errors = result.get("errors", [])
	if errors:
		logger.warning("Text embedding errors: %s", len(errors))

	store = vector_store_cls(dimension=service.dimension, metric="ip")
	store.add(vectors, metadata)

	filename = f"reference_text_{sanitize_model_name(service.model_name)}.faiss"
	index_path = output_dir / filename
	saved_index, saved_metadata = store.save(index_path)

	logger.info("Text index saved: %s", saved_index)
	logger.info("Text metadata saved: %s", saved_metadata)
	logger.info("Text vectors indexed: %s", store.size)
	logger.info("Text success count: %s", result.get("successful", 0))
	return saved_index


def create_image_index(
	products: list[Any],
	config_path: Path,
	output_dir: Path,
	image_timeout: float,
	image_batch_size: int,
	image_download_workers: int,
	image_process_workers: int,
	image_service_cls: Any,
	vector_store_cls: Any,
) -> Path:
	service = image_service_cls(config_path=config_path)
	vectors = []
	metadata = []
	candidates: list[tuple[Any, str]] = []

	for product in products:
		image_url = pick_main_image_url(product)
		if not image_url:
			logger.debug("Skipping sku=%s: missing image", product.sku)
			continue
		candidates.append((product, image_url))

	for start in range(0, len(candidates), image_batch_size):
		batch = candidates[start : start + image_batch_size]
		batch_urls = [item[1] for item in batch]

		try:
			batch_embeddings = service.embed_image_urls_batch(
				image_urls=batch_urls,
				batch_size=image_batch_size,
				download_workers=image_download_workers,
				process_workers=image_process_workers,
				timeout=image_timeout,
			)
			for (product, image_url), embedding in zip(batch, batch_embeddings, strict=False):
				if embedding is None:
					continue
				vectors.append(embedding)
				metadata.append(build_image_metadata(product, image_url=image_url))
		except Exception as exc:
			logger.warning("Image batch embedding failed (%s:%s): %s", start, start + len(batch), exc)
			for product, image_url in batch:
				try:
					vectors.append(service.embed_image_url(image_url=image_url, timeout=image_timeout))
					metadata.append(build_image_metadata(product, image_url=image_url))
				except Exception as item_exc:
					logger.warning("Image embedding failed for sku=%s: %s", product.sku, item_exc)

	store = vector_store_cls(dimension=service.dimension, metric="ip")
	store.add(vectors, metadata)

	filename = f"reference_image_{sanitize_model_name(service.model_name)}.faiss"
	index_path = output_dir / filename
	saved_index, saved_metadata = store.save(index_path)

	logger.info("Image index saved: %s", saved_index)
	logger.info("Image metadata saved: %s", saved_metadata)
	logger.info("Image vectors indexed: %s", store.size)
	return saved_index


def main() -> int:
	args = parse_args()
	(
		_reference_product_cls,
		reference_loader_cls,
		image_service_cls,
		text_service_cls,
		text_processor_cls,
		vector_store_cls,
	) = _resolve_symbols()

	production_config = _load_yaml(args.production_config)
	pipeline_config = production_config.get("pipeline", {}) if isinstance(production_config.get("pipeline", {}), dict) else {}

	text_batch_size = int(args.text_batch_size or pipeline_config.get("text_embedding_batch", 512))
	image_batch_size = int(args.image_batch_size or pipeline_config.get("image_embedding_batch", 64))
	image_download_workers = int(args.image_download_workers or pipeline_config.get("image_download_workers", 32))
	image_process_workers = int(args.image_process_workers or pipeline_config.get("image_process_workers", 8))

	logging.basicConfig(
		level=getattr(logging, args.log_level.upper(), logging.INFO),
		format="%(asctime)s | %(levelname)s | %(message)s",
	)

	embed_text = args.embed_text
	embed_images = args.embed_images
	if not embed_text and not embed_images:
		logger.info("No explicit mode selected, enabling both text and image indexing")
		embed_text = True
		embed_images = True

	loader = reference_loader_cls()
	products = loader.load(args.input_file)
	if args.limit is not None:
		products = products[: args.limit]

	logger.info("Loaded %s reference products", len(products))
	args.output_dir.mkdir(parents=True, exist_ok=True)

	if embed_text:
		create_text_index(
			products=products,
			config_path=args.config,
			output_dir=args.output_dir,
			text_service_cls=text_service_cls,
			text_processor_cls=text_processor_cls,
			vector_store_cls=vector_store_cls,
			batch_size=text_batch_size,
		)

	if embed_images:
		create_image_index(
			products=products,
			config_path=args.config,
			output_dir=args.output_dir,
			image_timeout=args.image_timeout,
			image_batch_size=image_batch_size,
			image_download_workers=image_download_workers,
			image_process_workers=image_process_workers,
			image_service_cls=image_service_cls,
			vector_store_cls=vector_store_cls,
		)

	logger.info("Reference indexing finished")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


"""Provider-agnostic image embedding service."""

from __future__ import annotations

import logging
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

try:
	import yaml
except ImportError as exc:  # pragma: no cover - dependency check
	raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


def _load_yaml(path: str | Path) -> dict[str, Any]:
	config_path = Path(path)
	payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		return {}
	return payload


class ImageEmbeddingService:
	def __init__(self, config_path: str | Path) -> None:
		self._config_path = Path(config_path)
		self._config = _load_yaml(self._config_path)
		self._embedder = self._build_embedder(self._config.get("image_embedding", {}))

	@property
	def model_name(self) -> str:
		return self._embedder.model_name

	@property
	def dimension(self) -> int:
		return int(self._embedder.dimension)

	def embed_image_path(self, image_path: str | Path):
		return self._embedder.embed_image(image_path)

	def embed_image_from_pil(self, image):
		return self._embedder.embed_image_from_pil(image)

	def embed_image_url(self, image_url: str, timeout: float = 20.0):
		image = self._download_image(image_url=image_url, timeout=timeout)
		if image is None:
			raise ValueError(f"Could not download image from URL: {image_url}")
		return self._embedder.embed_image_from_pil(image)

	def embed_image_urls_batch(
		self,
		image_urls: list[str],
		batch_size: int = 64,
		download_workers: int = 32,
		process_workers: int = 8,
		timeout: float = 20.0,
	):
		if not image_urls:
			return []

		images: list[Any | None] = [None] * len(image_urls)
		max_workers = max(1, int(download_workers))

		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			future_to_index = {
				executor.submit(self._download_image, url, timeout): idx
				for idx, url in enumerate(image_urls)
			}

			for future in as_completed(future_to_index):
				idx = future_to_index[future]
				try:
					images[idx] = future.result()
				except Exception as exc:
					logger.debug("Image download failed at index=%s: %s", idx, exc)

		if hasattr(self._embedder, "embed_pil_batch"):
			return self._embedder.embed_pil_batch(images, batch_size=batch_size)

		# Fallback for embedders without native batch support.
		results: list[Any | None] = [None] * len(images)
		if int(process_workers) > 1:
			with ThreadPoolExecutor(max_workers=int(process_workers)) as executor:
				future_to_index = {
					executor.submit(self._embedder.embed_image_from_pil, img): idx
					for idx, img in enumerate(images)
					if img is not None
				}
				for future in as_completed(future_to_index):
					idx = future_to_index[future]
					try:
						results[idx] = future.result()
					except Exception as exc:
						logger.debug("Image embedding failed at index=%s: %s", idx, exc)
			return results

		for idx, image in enumerate(images):
			if image is None:
				continue
			try:
				results[idx] = self._embedder.embed_image_from_pil(image)
			except Exception as exc:
				logger.debug("Image embedding failed at index=%s: %s", idx, exc)

		return results

	def _download_image(self, image_url: str, timeout: float = 20.0):
		try:
			import requests
		except ImportError as exc:
			raise ImportError(
				"requests and pillow are required for URL image embedding. Install with: pip install requests pillow"
			) from exc

		try:
			response = requests.get(image_url, timeout=timeout)
			response.raise_for_status()
			Image = importlib.import_module("PIL.Image")
			return Image.open(BytesIO(response.content)).convert("RGB")
		except Exception as exc:
			logger.debug("Failed to download image %s: %s", image_url, exc)
			return None

	def _build_embedder(self, config: dict[str, Any]):
		provider = str(config.get("provider", "dino")).lower()
		model = str(config.get("model", "vit_base_patch16_dinov3"))
		device = str(config.get("device", "auto"))

		if provider == "dino":
			try:
				from infra.embeddings_models.dino import DINOEmbedder
			except ModuleNotFoundError:
				from src.infra.embeddings_models.dino import DINOEmbedder

			return DINOEmbedder(model=model, device=device)

		if provider in {"fashion_clip", "fashion-clip"}:
			try:
				from infra.embeddings_models.fashion_clip import FashionCLIPEmbedder
			except ModuleNotFoundError:
				from src.infra.embeddings_models.fashion_clip import FashionCLIPEmbedder

			return FashionCLIPEmbedder(model=model, device=device)

		raise ValueError(f"Unsupported image embedding provider: {provider}")


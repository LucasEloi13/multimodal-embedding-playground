"""Provider-agnostic text embedding service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
	import yaml
except ImportError as exc:  # pragma: no cover - dependency check
	raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
	from core.entities.reference_product import ReferenceProduct
except ModuleNotFoundError:
	from src.core.entities.reference_product import ReferenceProduct

try:
	from core.utils.text_normalizer import TextNormalizer
except ModuleNotFoundError:
	from src.core.utils.text_normalizer import TextNormalizer


def _load_yaml(path: str | Path) -> dict[str, Any]:
	config_path = Path(path)
	payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		return {}
	return payload


class TextEmbeddingService:
	def __init__(self, config_path: str | Path) -> None:
		self._config_path = Path(config_path)
		self._config = _load_yaml(self._config_path)
		self._embedder = self._build_embedder(self._config.get("text_embedding", {}))
		self._text_normalizer = TextNormalizer()

	@property
	def model_name(self) -> str:
		return self._embedder.model_name

	@property
	def dimension(self) -> int:
		return int(self._embedder.dimension)

	def embed_text(self, text: str):
		return self._embedder.embed_text(text)

	def embed_texts_batch(self, texts: list[str], batch_size: int = 512):
		if hasattr(self._embedder, "embed_texts_batch"):
			return self._embedder.embed_texts_batch(texts, batch_size=batch_size)
		return [self._embedder.embed_text(text) for text in texts]

	def embed_reference_product(self, product: ReferenceProduct):
		text = self._text_normalizer.build_reference_product_sentence(product)
		return self._embedder.embed_text(text)

	def _build_embedder(self, config: dict[str, Any]):
		provider = str(config.get("provider", "openai")).lower()
		model = str(config.get("model", "text-embedding-3-large"))
		max_retries = int(config.get("max_retries", 5))
		initial_delay = float(config.get("initial_delay", 1.0))

		if provider == "openai":
			try:
				from infra.embeddings_models.openai_text import (
					OpenAITextEmbedder,
					load_dotenv_from_project_root,
				)
			except ModuleNotFoundError:
				from src.infra.embeddings_models.openai_text import (
					OpenAITextEmbedder,
					load_dotenv_from_project_root,
				)

			load_dotenv_from_project_root()
			return OpenAITextEmbedder(
				model=model,
				max_retries=max_retries,
				initial_delay=initial_delay,
			)

		raise ValueError(f"Unsupported text embedding provider: {provider}")


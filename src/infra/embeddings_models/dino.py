"""DINO image embedding provider based on timm models."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Conservative defaults that avoid OpenMP issues on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class DINOEmbedder:
	"""Generates normalized image embeddings using DINOv3 models from timm."""

	MODELS: dict[str, tuple[int, tuple[int, int]]] = {
		"vit_base_patch16_dinov3": (768, (518, 518)),
	}

	MODEL_ALIASES: dict[str, str] = {
		"dinov3": "vit_base_patch16_dinov3",
		"dinov3_base": "vit_base_patch16_dinov3",
	}

	def __init__(
		self,
		model: str = "vit_base_patch16_dinov3",
		device: str = "cpu",
		use_cls_token: bool = True,
	) -> None:
		self._torch = importlib.import_module("torch")
		self._use_cls_token = use_cls_token

		self._model_name = self.MODEL_ALIASES.get(model, model)
		self._dimension, self._input_size = self.MODELS.get(
			self._model_name,
			(768, (518, 518)),
		)
		self._device = self._resolve_device(device)
		self._load_timm_model(self._model_name)

		logger.info(
			"DINO loaded: model=%s dimension=%s device=%s",
			self._model_name,
			self._dimension,
			self._device,
		)

	@property
	def model_name(self) -> str:
		return f"dino-{self._model_name}"

	@property
	def dimension(self) -> int:
		return self._dimension

	@property
	def input_size(self) -> tuple[int, int]:
		return self._input_size

	@property
	def device_type(self) -> str:
		return self._device.type

	def embed_image(self, image_path: str | Path) -> np.ndarray:
		Image = importlib.import_module("PIL.Image")
		image = Image.open(image_path).convert("RGB")
		return self.embed_image_from_pil(image)

	def embed_image_from_pil(self, image: Any) -> np.ndarray:
		with self._torch.no_grad():
			image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
			features = self._model(image_tensor)
			embedding = self._extract_embedding(features)

		embedding = embedding.squeeze().cpu().numpy().astype("float32")
		if embedding.ndim > 1:
			embedding = embedding[0]

		norm = np.linalg.norm(embedding)
		if norm > 0:
			embedding = embedding / norm

		return embedding

	def embed_pil_batch(
		self,
		pil_images: list[Any],
		batch_size: int = 64,
	) -> list[np.ndarray | None]:
		results: list[np.ndarray | None] = [None] * len(pil_images)
		prepared: list[tuple[int, Any]] = []

		for index, image in enumerate(pil_images):
			if image is None:
				continue
			try:
				image_rgb = image.convert("RGB") if image.mode != "RGB" else image
				prepared.append((index, self._preprocess(image_rgb)))
			except Exception as exc:  # pragma: no cover - defensive path
				logger.debug("Failed preprocessing image index %s: %s", index, exc)

		if not prepared:
			return results

		for start in range(0, len(prepared), batch_size):
			chunk = prepared[start : start + batch_size]
			indices = [item[0] for item in chunk]
			tensors = [item[1] for item in chunk]
			batch = self._torch.stack(tensors).to(self._device)

			with self._torch.no_grad():
				features = self._model(batch)
				embeddings = self._extract_embedding(features)

			embeddings = embeddings.cpu().numpy().astype("float32")
			norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
			norms = np.where(norms > 0, norms, 1)
			embeddings = embeddings / norms

			for idx, emb in zip(indices, embeddings, strict=False):
				results[idx] = emb

		self._clear_cache()
		return results

	@classmethod
	def list_available_models(cls) -> list[str]:
		return list(cls.MODELS.keys()) + list(cls.MODEL_ALIASES.keys())

	def _resolve_device(self, device: str):
		torch = self._torch
		if device == "auto":
			if torch.cuda.is_available():
				return torch.device("cuda")
			if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
				return torch.device("mps")
			return torch.device("cpu")

		if device == "cuda" and not torch.cuda.is_available():
			if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
				logger.warning("CUDA unavailable, falling back to MPS")
				return torch.device("mps")
			logger.warning("CUDA unavailable, falling back to CPU")
			return torch.device("cpu")

		return torch.device(device)

	def _load_timm_model(self, model: str) -> None:
		try:
			timm = importlib.import_module("timm")
			resolve_data_config = importlib.import_module("timm.data").resolve_data_config
			create_transform = importlib.import_module(
				"timm.data.transforms_factory"
			).create_transform
		except ImportError as exc:
			raise ImportError(
				"timm is required for DINOEmbedder. Install with: pip install timm torch pillow"
			) from exc

		self._model = timm.create_model(model, pretrained=True, num_classes=0)
		self._model = self._model.to(self._device)
		self._model.eval()

		if hasattr(self._model, "num_features"):
			self._dimension = int(self._model.num_features)
		elif hasattr(self._model, "embed_dim"):
			self._dimension = int(self._model.embed_dim)

		data_config = resolve_data_config(self._model.pretrained_cfg)
		self._preprocess = create_transform(**data_config, is_training=False)

		input_size = data_config.get("input_size")
		if isinstance(input_size, (tuple, list)) and len(input_size) >= 2:
			self._input_size = (int(input_size[-2]), int(input_size[-1]))

	def _extract_embedding(self, features: Any):
		if isinstance(features, dict):
			if "x_norm_clstoken" in features:
				return features["x_norm_clstoken"]
			if "x_prenorm" in features:
				return features["x_prenorm"][:, 0]
			candidate = list(features.values())[0]
			return candidate[:, 0] if getattr(candidate, "ndim", 0) == 3 else candidate

		if getattr(features, "ndim", 0) == 3:
			if self._use_cls_token:
				return features[:, 0, :]
			return features.mean(dim=1)

		return features

	def _clear_cache(self) -> None:
		try:
			if self._device.type == "cuda":
				self._torch.cuda.empty_cache()
			elif self._device.type == "mps":
				self._torch.mps.empty_cache()
		except Exception:
			pass

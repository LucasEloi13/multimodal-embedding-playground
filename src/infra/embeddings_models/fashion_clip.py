"""Fashion CLIP image embedding provider."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from core.utils.image_preprocessing import build_transform, preprocess_pil_image
except ModuleNotFoundError:
    from src.core.utils.image_preprocessing import build_transform, preprocess_pil_image

logger = logging.getLogger(__name__)

# macOS/OpenMP safety guards to avoid duplicate libomp initialization crashes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class FashionCLIPEmbedder:
    """Generates normalized image embeddings using Fashion CLIP."""

    MODEL_ALIASES: dict[str, str] = {
        "fashion_clip": "patrickjohncyh/fashion-clip",
        "fashion-clip": "patrickjohncyh/fashion-clip",
    }

    def __init__(
        self,
        model: str = "patrickjohncyh/fashion-clip",
        device: str = "auto",
    ) -> None:
        self._torch = importlib.import_module("torch")

        resolved_model = self.MODEL_ALIASES.get(model, model)
        self._model_id = resolved_model
        self._device = self._resolve_device(device)
        self._transform = build_transform(augment=False)

        self._load_model()

        logger.info(
            "Fashion CLIP loaded: model=%s dimension=%s device=%s",
            self._model_id,
            self._dimension,
            self._device,
        )

    @property
    def model_name(self) -> str:
        safe_name = self._model_id.replace("/", "-")
        return f"fashion-clip-{safe_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        Image = importlib.import_module("PIL.Image")
        image = Image.open(image_path)
        return self.embed_image_from_pil(image)

    def embed_image_from_pil(self, image: Any) -> np.ndarray:
        batch_embeddings = self.embed_pil_batch([image], batch_size=1)
        embedding = batch_embeddings[0]
        if embedding is None:
            raise ValueError("Could not generate image embedding")
        return embedding

    def embed_pil_batch(self, pil_images: list[Any], batch_size: int = 64) -> list[np.ndarray | None]:
        results: list[np.ndarray | None] = [None] * len(pil_images)

        valid: list[tuple[int, Any]] = []
        for idx, image in enumerate(pil_images):
            if image is None:
                continue
            try:
                pixel_values = preprocess_pil_image(
                    image=image,
                    transform=self._transform,
                    remove_background=False,
                )
                if pixel_values is None:
                    continue
                valid.append((idx, pixel_values))
            except Exception as exc:
                logger.debug("Image preprocessing failed at index=%s: %s", idx, exc)

        if not valid:
            return results

        for start in range(0, len(valid), max(1, int(batch_size))):
            chunk = valid[start : start + max(1, int(batch_size))]
            indices = [item[0] for item in chunk]
            tensors = [item[1].squeeze(0) for item in chunk]

            with self._torch.no_grad():
                batch_pixel_values = self._torch.stack(tensors).to(self._device)
                raw_output = self._model.get_image_features(pixel_values=batch_pixel_values)
                embeddings = self._extract_embeddings(raw_output)

            embeddings = embeddings.detach().cpu().numpy().astype("float32")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms

            for idx, emb in zip(indices, embeddings, strict=False):
                results[idx] = emb

        self._clear_cache()
        return results

    def _load_model(self) -> None:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "transformers is required for FashionCLIPEmbedder. Install with: pip install transformers"
            ) from exc

        model_cls = getattr(transformers, "CLIPModel")
        self._model = model_cls.from_pretrained(self._model_id)
        self._model = self._model.to(self._device)
        self._model.eval()

        self._dimension = int(getattr(self._model.config, "projection_dim", 512))

    def _extract_embeddings(self, output: Any):
        # Most CLIP implementations return a tensor directly.
        if self._torch.is_tensor(output):
            return output

        # Some variants return structured outputs.
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output

        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            hidden = output.last_hidden_state
            if getattr(hidden, "ndim", 0) == 3:
                return hidden[:, 0, :]
            return hidden

        if isinstance(output, dict):
            for key in ("image_embeds", "pooler_output", "last_hidden_state"):
                value = output.get(key)
                if value is None:
                    continue
                if getattr(value, "ndim", 0) == 3:
                    return value[:, 0, :]
                return value

            if output:
                first = next(iter(output.values()))
                if getattr(first, "ndim", 0) == 3:
                    return first[:, 0, :]
                return first

        raise TypeError(f"Unsupported model output type for embeddings: {type(output)!r}")

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

    def _clear_cache(self) -> None:
        try:
            if self._device.type == "cuda":
                self._torch.cuda.empty_cache()
            elif self._device.type == "mps":
                self._torch.mps.empty_cache()
        except Exception:
            pass

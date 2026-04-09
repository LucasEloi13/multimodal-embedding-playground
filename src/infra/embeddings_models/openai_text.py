"""OpenAI text embedding provider."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


class OpenAITextEmbedder:
    MODELS: dict[str, int] = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ) -> None:
        self._model = model
        self._dimension = self.MODELS.get(model, 3072)
        self._max_retries = max_retries
        self._initial_delay = initial_delay

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set environment variable or pass api_key explicitly."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=resolved_key)

    @property
    def model_name(self) -> str:
        return f"openai-{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        return self._create_embedding_with_retry(text)

    def embed_texts_batch(self, texts: list[str], batch_size: int = 2048) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            def _create() -> list[np.ndarray]:
                response = self._client.embeddings.create(model=self._model, input=batch)
                return [
                    np.array(item.embedding, dtype="float32")
                    for item in response.data
                ]

            embeddings.extend(self._retry_with_backoff(_create))

        return embeddings

    def embed_structured(self, payload: dict[str, Any]) -> np.ndarray:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return self._create_embedding_with_retry(text)

    def _create_embedding_with_retry(self, text: str) -> np.ndarray:
        def _create() -> np.ndarray:
            response = self._client.embeddings.create(model=self._model, input=text)
            return np.array(response.data[0].embedding, dtype="float32")

        return self._retry_with_backoff(_create)

    def _retry_with_backoff(self, func):
        delay = self._initial_delay
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                return func()
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_error = exc
                if attempt == self._max_retries - 1:
                    break

                error_msg = str(exc).lower()
                wait_time = delay * 2 if "rate" in error_msg or "429" in error_msg else delay
                time.sleep(wait_time)
                delay *= 2

        raise RuntimeError("Failed to generate embedding after retries") from last_error


def load_dotenv_from_project_root() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

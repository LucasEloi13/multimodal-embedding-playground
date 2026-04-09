"""FAISS-backed vector store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FaissVectorStore:
	def __init__(self, dimension: int, metric: str = "ip") -> None:
		self._dimension = int(dimension)
		self._metric = metric.lower()
		self._metadata: list[dict[str, Any]] = []

		try:
			import faiss
		except ImportError as exc:
			raise ImportError(
				"faiss is required. Install with: pip install faiss-cpu"
			) from exc

		self._faiss = faiss
		if self._metric == "l2":
			self._index = faiss.IndexFlatL2(self._dimension)
		else:
			self._index = faiss.IndexFlatIP(self._dimension)

	@property
	def dimension(self) -> int:
		return self._dimension

	@property
	def size(self) -> int:
		return int(self._index.ntotal)

	def add(self, vectors: list[np.ndarray], metadata: list[dict[str, Any]] | None = None) -> None:
		if not vectors:
			return

		matrix = np.vstack([np.asarray(vec, dtype="float32") for vec in vectors])
		if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
			raise ValueError(
				f"Expected vectors with shape (*, {self._dimension}), got {matrix.shape}"
			)

		self._index.add(matrix)
		if metadata:
			self._metadata.extend(metadata)

	def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
		query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
		distances, indices = self._index.search(query, top_k)

		results: list[dict[str, Any]] = []
		for score, idx in zip(distances[0], indices[0], strict=False):
			if idx < 0:
				continue
			payload = self._metadata[idx] if idx < len(self._metadata) else {}
			results.append({"score": float(score), "index": int(idx), "metadata": payload})

		return results

	def find_indices_by_metadata(self, key: str, value: Any) -> list[int]:
		indices: list[int] = []
		for idx, payload in enumerate(self._metadata):
			if payload.get(key) == value:
				indices.append(idx)
		return indices

	def get_vector(self, index: int) -> np.ndarray:
		if index < 0 or index >= self.size:
			raise IndexError(f"Vector index out of range: {index}")
		vector = self._index.reconstruct(int(index))
		return np.asarray(vector, dtype="float32")

	def save(self, index_path: str | Path, metadata_path: str | Path | None = None) -> tuple[Path, Path]:
		index_output = Path(index_path)
		index_output.parent.mkdir(parents=True, exist_ok=True)

		self._faiss.write_index(self._index, str(index_output))

		metadata_output = (
			Path(metadata_path)
			if metadata_path is not None
			else index_output.with_suffix(index_output.suffix + ".meta.jsonl")
		)
		metadata_output.parent.mkdir(parents=True, exist_ok=True)

		with metadata_output.open("w", encoding="utf-8") as handle:
			for item in self._metadata:
				handle.write(json.dumps(item, ensure_ascii=False) + "\n")

		return index_output, metadata_output

	@classmethod
	def from_files(
		cls,
		index_path: str | Path,
		metadata_path: str | Path | None = None,
		metric: str = "ip",
	) -> "FaissVectorStore":
		index_input = Path(index_path)
		if not index_input.exists():
			raise FileNotFoundError(f"Index file not found: {index_input}")

		try:
			import faiss
		except ImportError as exc:
			raise ImportError(
				"faiss is required. Install with: pip install faiss-cpu"
			) from exc

		index = faiss.read_index(str(index_input))
		store = cls(dimension=int(index.d), metric=metric)
		store._index = index

		meta_file = (
			Path(metadata_path)
			if metadata_path is not None
			else index_input.with_suffix(index_input.suffix + ".meta.jsonl")
		)
		if meta_file.exists():
			store._metadata = cls._read_metadata(meta_file)

		return store

	@staticmethod
	def _read_metadata(metadata_path: Path) -> list[dict[str, Any]]:
		metadata: list[dict[str, Any]] = []
		with metadata_path.open("r", encoding="utf-8") as handle:
			for raw_line in handle:
				line = raw_line.strip()
				if not line:
					continue
				try:
					payload = json.loads(line)
				except json.JSONDecodeError:
					continue
				if isinstance(payload, dict):
					metadata.append(payload)

		return metadata


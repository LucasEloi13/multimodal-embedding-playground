"""Prepare query/top1 pairs into a compact LLM-friendly comparison payload."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class ProductForLLM:
	role: str
	sku: str | None
	name: str
	brand: str
	description: str
	image_url: str | None
	attributes: dict[str, str]

	def to_dict(self) -> dict[str, Any]:
		return {
			"role": self.role,
			"sku": self.sku,
			"name": self.name,
			"brand": self.brand,
			"description": self.description,
			"image_url": self.image_url,
			"attributes": self.attributes,
		}


@dataclass
class ProductComparisonItem:
	index: int
	query: ProductForLLM
	candidate: ProductForLLM

	def to_payload(self) -> dict[str, Any]:
		return {
			"pair_id": self.index,
			"query": self.query.to_dict(),
			"candidate": self.candidate.to_dict(),
		}


class LLMEvaluationProcessor:
	def load_comparisons(self, input_path: str | Path, limit: int | None = None) -> list[ProductComparisonItem]:
		path = Path(input_path)
		rows = json.loads(path.read_text(encoding="utf-8"))
		if not isinstance(rows, list):
			raise ValueError(f"Expected JSON list in {path}")

		comparisons: list[ProductComparisonItem] = []
		for idx, row in enumerate(rows):
			if limit is not None and len(comparisons) >= limit:
				break

			query_raw = row.get("query") if isinstance(row, dict) else None
			candidate_raw = row.get("top1_candidate") if isinstance(row, dict) else None
			if not isinstance(query_raw, dict) or not isinstance(candidate_raw, dict):
				continue

			query = self._build_product(role="query", payload=query_raw, sku_key="sku")
			candidate = self._build_product(role="candidate", payload=candidate_raw, sku_key="candidate_sku")
			comparisons.append(ProductComparisonItem(index=idx, query=query, candidate=candidate))

		return comparisons

	def _build_product(self, role: str, payload: dict[str, Any], sku_key: str) -> ProductForLLM:
		return ProductForLLM(
			role=role,
			sku=self._as_optional_string(payload.get(sku_key)),
			name=self._as_string(payload.get("name")),
			brand=self._as_string(payload.get("brand")),
			description=self._as_string(payload.get("description")),
			image_url=self._as_optional_string(payload.get("image_url")),
			attributes=self._normalize_attributes(payload.get("additional_properties")),
		)

	def _normalize_attributes(self, value: Any) -> dict[str, str]:
		if not isinstance(value, list):
			return {}

		attrs: dict[str, str] = {}
		for item in value:
			if not isinstance(item, dict):
				continue
			name = self._as_optional_string(item.get("name"))
			attr_value = self._as_optional_string(item.get("value"))
			if not name or not attr_value:
				continue
			if name not in attrs:
				attrs[name] = attr_value
		return attrs

	def _as_string(self, value: Any) -> str:
		return str(value) if value is not None else ""

	def _as_optional_string(self, value: Any) -> str | None:
		if value is None:
			return None
		text = str(value).strip()
		return text or None

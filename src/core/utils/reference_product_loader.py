from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

try:
	from core.entities.reference_product import ReferenceProduct
except ModuleNotFoundError:  # pragma: no cover - fallback for alternate PYTHONPATH setups
	from src.core.entities.reference_product import ReferenceProduct


class UnsupportedReferenceProductFormatError(ValueError):
	pass


class InvalidReferenceProductPayloadError(ValueError):
	pass


class ReferenceProductParser(Protocol):
	def parse(self, raw_text: str) -> list[dict[str, Any]]:
		...


class JsonReferenceProductParser:
	"""Parser for .json, .jsonl and ndjson payloads."""

	def parse(self, raw_text: str) -> list[dict[str, Any]]:
		text = raw_text.strip()
		if not text:
			return []

		parsed = self._try_parse_as_json(text)
		if parsed is not None:
			return self._normalize(parsed)

		return self._parse_as_jsonl(text)

	def _try_parse_as_json(self, text: str) -> Any | None:
		try:
			return json.loads(text)
		except json.JSONDecodeError:
			return None

	def _parse_as_jsonl(self, text: str) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for line_number, raw_line in enumerate(text.splitlines(), start=1):
			line = raw_line.strip()
			if not line:
				continue

			try:
				payload = json.loads(line)
			except json.JSONDecodeError as exc:
				raise InvalidReferenceProductPayloadError(
					f"Invalid JSONL at line {line_number}: {exc.msg}"
				) from exc

			records.extend(self._normalize(payload))

		return records

	def _normalize(self, payload: Any) -> list[dict[str, Any]]:
		if isinstance(payload, list):
			return [item for item in payload if isinstance(item, dict)]

		if isinstance(payload, dict):
			for container_key in ("items", "products", "data", "results"):
				container_value = payload.get(container_key)
				if isinstance(container_value, list):
					return [item for item in container_value if isinstance(item, dict)]
				if isinstance(container_value, dict):
					return [container_value]

			return [payload]

		raise InvalidReferenceProductPayloadError(
			"Payload must be a JSON object, JSON array or JSONL of objects"
		)


class ReferenceProductLoader:
	"""Load ReferenceProduct entities from different serialized formats."""

	def __init__(self, parsers: dict[str, ReferenceProductParser] | None = None) -> None:
		self._parsers: dict[str, ReferenceProductParser] = {}
		self.register_parser("json", JsonReferenceProductParser())
		self.register_parser("jsonl", JsonReferenceProductParser())
		self.register_parser("ndjson", JsonReferenceProductParser())

		if parsers:
			for format_name, parser in parsers.items():
				self.register_parser(format_name, parser)

	def register_parser(self, format_name: str, parser: ReferenceProductParser) -> None:
		normalized_format = format_name.lower().lstrip(".")
		self._parsers[normalized_format] = parser

	def load(
		self,
		file_path: str | Path,
		format_name: str | None = None,
	) -> list[ReferenceProduct]:
		path = Path(file_path)
		raw_text = path.read_text(encoding="utf-8")

		parser = self._resolve_parser(path=path, format_name=format_name)
		records = parser.parse(raw_text)

		return [ReferenceProduct.from_dict(record) for record in records]

	def _resolve_parser(
		self,
		path: Path,
		format_name: str | None,
	) -> ReferenceProductParser:
		normalized_format = (format_name or path.suffix).lower().lstrip(".")
		parser = self._parsers.get(normalized_format)
		if parser is None:
			supported_formats = ", ".join(sorted(self._parsers.keys()))
			raise UnsupportedReferenceProductFormatError(
				f"Unsupported format '{normalized_format}'. Supported: {supported_formats}"
			)

		return parser

"""Gemini provider for product-vs-product comparison using text and images."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests


class GeminiLLMProvider:
	def __init__(
		self,
		model: str,
		prompt_text: str,
		temperature: float = 0.0,
		request_timeout_seconds: float = 30.0,
	) -> None:
		load_dotenv(dotenv_path=Path.cwd() / ".env")
		load_dotenv()
		self.model = model
		self.prompt_text = prompt_text
		self.temperature = temperature
		self.request_timeout_seconds = request_timeout_seconds
		self._client = genai.Client()

	def evaluate_pair(self, payload: dict[str, Any]) -> dict[str, Any]:
		query = payload.get("query", {}) if isinstance(payload.get("query", {}), dict) else {}
		candidate = payload.get("candidate", {}) if isinstance(payload.get("candidate", {}), dict) else {}

		query_image = self._build_image_part(query.get("image_url"))
		candidate_image = self._build_image_part(candidate.get("image_url"))

		text_payload = json.dumps(payload, ensure_ascii=False, indent=2)
		contents: list[Any] = [
			self.prompt_text,
			"Dados estruturados do par para comparacao:",
			text_payload,
			"Imagem do produto QUERY:",
		]
		if query_image is not None:
			contents.append(query_image)
		else:
			contents.append("(sem imagem QUERY)")

		contents.append("Imagem do produto CANDIDATE:")
		if candidate_image is not None:
			contents.append(candidate_image)
		else:
			contents.append("(sem imagem CANDIDATE)")

		response = self._client.models.generate_content(
			model=self.model,
			contents=contents,
			config=types.GenerateContentConfig(temperature=self.temperature),
		)

		raw_text = response.text or ""
		parsed = self._parse_response_json(raw_text)
		usage = self._extract_usage(response)

		return {
			"provider": "gemini",
			"model": self.model,
			"raw_response": raw_text,
			"result": parsed,
			"usage": usage,
		}

	def _build_image_part(self, image_url: Any) -> types.Part | None:
		if not isinstance(image_url, str) or not image_url.strip():
			return None

		response = requests.get(image_url, timeout=self.request_timeout_seconds)
		response.raise_for_status()
		image_bytes = response.content
		content_type = response.headers.get("content-type")
		mime_type = self._resolve_mime_type(content_type=content_type, image_url=image_url)
		return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

	def _resolve_mime_type(self, content_type: str | None, image_url: str) -> str:
		if content_type:
			mime = content_type.split(";")[0].strip().lower()
			if mime.startswith("image/"):
				return mime

		guessed, _ = mimetypes.guess_type(image_url)
		if guessed and guessed.startswith("image/"):
			return guessed
		return "image/jpeg"

	def _extract_usage(self, response: Any) -> dict[str, int | None]:
		usage = getattr(response, "usage_metadata", None)
		if usage is None:
			return {
				"input_tokens": None,
				"input_image_tokens": None,
				"input_total_tokens": None,
				"thoughts_tokens": None,
				"cached_content_tokens": None,
				"tool_use_prompt_tokens": None,
				"output_tokens": None,
				"accounted_total_tokens": None,
				"unaccounted_tokens": None,
				"total_tokens": None,
			}

		prompt_total_tokens = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_token_count", None)
		thoughts_tokens = getattr(usage, "thoughts_token_count", None)
		cached_content_tokens = getattr(usage, "cached_content_token_count", None)
		tool_use_prompt_tokens = getattr(usage, "tool_use_prompt_token_count", None)
		output_tokens = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", None)
		total_tokens = getattr(usage, "total_token_count", None)

		input_text_tokens, input_image_tokens = self._extract_prompt_modality_tokens(usage)
		if input_text_tokens is None and prompt_total_tokens is not None:
			# Fallback when detailed modality split is unavailable.
			input_text_tokens = prompt_total_tokens
			input_image_tokens = 0

		accounted_total_tokens = None
		unaccounted_tokens = None
		if prompt_total_tokens is not None and output_tokens is not None:
			accounted_total_tokens = int(prompt_total_tokens) + int(output_tokens) + int(thoughts_tokens or 0)
		if total_tokens is not None and accounted_total_tokens is not None:
			unaccounted_tokens = int(total_tokens) - int(accounted_total_tokens)

		return {
			"input_tokens": input_text_tokens,
			"input_image_tokens": input_image_tokens,
			"input_total_tokens": prompt_total_tokens,
			"thoughts_tokens": thoughts_tokens,
			"cached_content_tokens": cached_content_tokens,
			"tool_use_prompt_tokens": tool_use_prompt_tokens,
			"output_tokens": output_tokens,
			"accounted_total_tokens": accounted_total_tokens,
			"unaccounted_tokens": unaccounted_tokens,
			"total_tokens": total_tokens,
		}

	def _extract_prompt_modality_tokens(self, usage: Any) -> tuple[int | None, int | None]:
		details = getattr(usage, "prompt_tokens_details", None)
		if not isinstance(details, list):
			return None, None

		text_tokens = 0
		image_tokens = 0
		saw_any = False
		for item in details:
			modality = str(getattr(item, "modality", "")).upper()
			token_count = getattr(item, "token_count", None)
			if token_count is None:
				continue
			try:
				count = int(token_count)
			except (TypeError, ValueError):
				continue

			saw_any = True
			if "IMAGE" in modality:
				image_tokens += count
			elif "TEXT" in modality:
				text_tokens += count
			else:
				text_tokens += count

		if not saw_any:
			return None, None
		return text_tokens, image_tokens

	def _parse_response_json(self, text: str) -> dict[str, Any]:
		cleaned = text.strip()
		fenced_match = re.search(r"```(?:json)?\\s*(\{.*\}|\[.*\])\\s*```", cleaned, flags=re.DOTALL)
		if fenced_match:
			cleaned = fenced_match.group(1)

		try:
			parsed = json.loads(cleaned)
			if isinstance(parsed, dict):
				return parsed
			return {"parsed_payload": parsed}
		except json.JSONDecodeError:
			return {
				"decision": "unknown",
				"confidence": 0.0,
				"reasoning": "Model did not return valid JSON.",
			}

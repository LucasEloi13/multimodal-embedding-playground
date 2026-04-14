"""Service that orchestrates LLM-based comparison for query vs top1 candidate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import threading
from typing import Any

try:
	import yaml
except ImportError as exc:  # pragma: no cover - dependency check
	raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
	from embedding.services.llm_evaluation_processor import LLMEvaluationProcessor
except ModuleNotFoundError:
	from src.embedding.services.llm_evaluation_processor import LLMEvaluationProcessor


class LLMEvaluationService:
	def __init__(self, production_config_path: str | Path) -> None:
		self.production_config_path = Path(production_config_path)
		self.project_root = self.production_config_path.parents[3]
		self.config = self._load_yaml(self.production_config_path)

		llm_cfg = self.config.get("llm_evaluation", {})
		if not isinstance(llm_cfg, dict):
			llm_cfg = {}

		self.provider_name = str(llm_cfg.get("provider", "gemini")).lower()
		self.default_workers = int(llm_cfg.get("workers", 1))
		prompt_relative = str(llm_cfg.get("prompt_path", "src/infra/llm/prompt/prompt.md"))
		self.prompt_path = self.project_root / prompt_relative
		self.prompt_text = self.prompt_path.read_text(encoding="utf-8")

		provider_cfg = llm_cfg.get(self.provider_name, {})
		if not isinstance(provider_cfg, dict):
			provider_cfg = {}

		self.provider_cfg = provider_cfg
		self.provider = self._build_provider(self.provider_name, self.provider_cfg)
		self._provider_local = threading.local()
		self.processor = LLMEvaluationProcessor()

		usage_dir_rel = str(llm_cfg.get("usage_logs_dir", "src/infra/llm/usage_logs"))
		self.usage_logs_dir = self.project_root / usage_dir_rel
		self.usage_logs_dir.mkdir(parents=True, exist_ok=True)

	def evaluate_file(
		self,
		input_path: str | Path,
		output_path: str | Path,
		limit: int | None = None,
		workers: int | None = None,
	) -> dict[str, Any]:
		comparisons = self.processor.load_comparisons(input_path=input_path, limit=limit)
		deduped_comparisons = self._deduplicate_comparisons(comparisons)
		worker_count = max(1, int(workers or self.default_workers or 1))
		total_pairs = len(deduped_comparisons)

		run_started_at = datetime.now(timezone.utc)
		run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
		usage_run_file = self.usage_logs_dir / f"usage_{run_id}.jsonl"
		usage_totals_file = self.usage_logs_dir / "usage_totals.json"

		results: list[dict[str, Any]] = []
		run_input_tokens = 0
		run_input_image_tokens = 0
		run_input_total_tokens = 0
		run_thoughts_tokens = 0
		run_cached_content_tokens = 0
		run_tool_use_prompt_tokens = 0
		run_output_tokens = 0
		run_accounted_total_tokens = 0
		run_unaccounted_tokens = 0
		run_total_tokens = 0

		if worker_count == 1:
			evaluation_rows = []
			for idx, item in enumerate(deduped_comparisons, start=1):
				evaluation_rows.append(self._evaluate_single(item))
				self._print_progress(current=idx, total=total_pairs)
		else:
			evaluation_rows = []
			with ThreadPoolExecutor(max_workers=worker_count) as executor:
				futures = [executor.submit(self._evaluate_single, item) for item in deduped_comparisons]
				for completed, future in enumerate(as_completed(futures), start=1):
					evaluation_rows.append(future.result())
					self._print_progress(current=completed, total=total_pairs)

			evaluation_rows.sort(key=lambda row: row["pair_id"])

		if total_pairs > 0:
			sys.stderr.write("\n")
			sys.stderr.flush()

		usage_events: list[dict[str, Any]] = []
		for row in evaluation_rows:
			usage_event = row["usage"]
			usage_events.append(usage_event)
			run_input_tokens += int(usage_event.get("input_tokens") or 0)
			run_input_image_tokens += int(usage_event.get("input_image_tokens") or 0)
			run_input_total_tokens += int(usage_event.get("input_total_tokens") or 0)
			run_thoughts_tokens += int(usage_event.get("thoughts_tokens") or 0)
			run_cached_content_tokens += int(usage_event.get("cached_content_tokens") or 0)
			run_tool_use_prompt_tokens += int(usage_event.get("tool_use_prompt_tokens") or 0)
			run_output_tokens += int(usage_event.get("output_tokens") or 0)
			run_accounted_total_tokens += int(usage_event.get("accounted_total_tokens") or 0)
			run_unaccounted_tokens += int(usage_event.get("unaccounted_tokens") or 0)
			run_total_tokens += int(usage_event.get("total_tokens") or 0)

		for usage_event in usage_events:
			self._append_jsonl(usage_run_file, usage_event)

		self._write_structured_usage_json(
			source_jsonl_file=usage_run_file,
			run_id=run_id,
			run_started_at=run_started_at,
			usage_events=usage_events,
		)

		results = evaluation_rows

		self._update_usage_totals(
			totals_file=usage_totals_file,
			run_id=run_id,
			run_started_at=run_started_at,
			calls=len(results),
			input_tokens=run_input_tokens,
			input_image_tokens=run_input_image_tokens,
			input_total_tokens=run_input_total_tokens,
			thoughts_tokens=run_thoughts_tokens,
			cached_content_tokens=run_cached_content_tokens,
			tool_use_prompt_tokens=run_tool_use_prompt_tokens,
			output_tokens=run_output_tokens,
			accounted_total_tokens=run_accounted_total_tokens,
			unaccounted_tokens=run_unaccounted_tokens,
			total_tokens=run_total_tokens,
		)

		output_payload = {
			"metadata": {
				"provider": self.provider_name,
				"input_file": str(input_path),
				"items_evaluated": len(results),
				"items_received": len(comparisons),
				"items_after_dedup": len(deduped_comparisons),
				"run_id": run_id,
				"workers": worker_count,
				"usage": {
					"input_tokens": run_input_tokens,
					"input_image_tokens": run_input_image_tokens,
					"input_total_tokens": run_input_total_tokens,
					"thoughts_tokens": run_thoughts_tokens,
					"cached_content_tokens": run_cached_content_tokens,
					"tool_use_prompt_tokens": run_tool_use_prompt_tokens,
					"output_tokens": run_output_tokens,
					"accounted_total_tokens": run_accounted_total_tokens,
					"unaccounted_tokens": run_unaccounted_tokens,
					"total_tokens": run_total_tokens,
				},
			},
			"results": results,
		}

		output_file = Path(output_path)
		output_file.parent.mkdir(parents=True, exist_ok=True)
		output_file.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
		return output_payload

	def _build_provider(self, provider_name: str, provider_cfg: dict[str, Any]) -> Any:
		model = str(provider_cfg.get("model", ""))
		if not model:
			raise ValueError(f"Missing model for provider '{provider_name}' in production.yml")

		temperature = float(provider_cfg.get("temperature", 0.0))
		timeout = float(provider_cfg.get("request_timeout_seconds", 30.0))

		module_path = f"infra.llm.providers.{provider_name}"
		default_class_name = f"{provider_name.title()}LLMProvider"
		class_name = str(provider_cfg.get("class_name", default_class_name))

		cls = self._load_provider_class(module_path=module_path, class_name=class_name)
		return cls(
			model=model,
			prompt_text=self.prompt_text,
			temperature=temperature,
			request_timeout_seconds=timeout,
		)

	def _get_thread_provider(self) -> Any:
		provider = getattr(self._provider_local, "provider", None)
		if provider is None:
			provider = self._build_provider(self.provider_name, self.provider_cfg)
			self._provider_local.provider = provider
		return provider

	def _evaluate_single(self, item: Any) -> dict[str, Any]:
		payload = item.to_payload()
		provider = self._get_thread_provider()
		provider_result = provider.evaluate_pair(payload)

		usage = provider_result.get("usage", {}) if isinstance(provider_result.get("usage", {}), dict) else {}
		usage_event = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"provider": self.provider_name,
			"model": provider_result.get("model"),
			"query_sku": item.query.sku,
			"candidate_sku": item.candidate.sku,
			"input_tokens": int(usage.get("input_tokens") or 0),
			"input_image_tokens": int(usage.get("input_image_tokens") or 0),
			"input_total_tokens": int(usage.get("input_total_tokens") or 0),
			"thoughts_tokens": int(usage.get("thoughts_tokens") or 0),
			"cached_content_tokens": int(usage.get("cached_content_tokens") or 0),
			"tool_use_prompt_tokens": int(usage.get("tool_use_prompt_tokens") or 0),
			"output_tokens": int(usage.get("output_tokens") or 0),
			"accounted_total_tokens": int(usage.get("accounted_total_tokens") or 0),
			"unaccounted_tokens": int(usage.get("unaccounted_tokens") or 0),
			"total_tokens": int(usage.get("total_tokens") or 0),
		}

		return {
			"pair_id": payload.get("pair_id"),
			"query": payload.get("query"),
			"candidate": payload.get("candidate"),
			"llm_result": provider_result.get("result"),
			"usage": usage_event,
		}

	def _deduplicate_comparisons(self, comparisons: list[Any]) -> list[Any]:
		seen: set[tuple[str, str]] = set()
		deduped: list[Any] = []

		for item in comparisons:
			query_key = item.query.sku or item.query.image_url or ""
			candidate_key = item.candidate.sku or item.candidate.image_url or ""
			pair_key = (query_key.strip().lower(), candidate_key.strip().lower())
			if pair_key in seen:
				continue
			seen.add(pair_key)
			deduped.append(item)

		return deduped

	def _load_provider_class(self, module_path: str, class_name: str) -> Any:
		try:
			module = importlib.import_module(module_path)
		except ModuleNotFoundError:
			module = importlib.import_module(f"src.{module_path}")
		return getattr(module, class_name)

	def _print_progress(self, current: int, total: int, width: int = 28) -> None:
		if total <= 0:
			return
		ratio = min(1.0, max(0.0, current / total))
		filled = int(width * ratio)
		bar = "#" * filled + "-" * (width - filled)
		sys.stderr.write(f"\rProgress [{bar}] {current}/{total}")
		sys.stderr.flush()

	def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
		with path.open("a", encoding="utf-8") as f:
			f.write(json.dumps(payload, ensure_ascii=False) + "\\n")

	def _write_structured_usage_json(
		self,
		source_jsonl_file: Path,
		run_id: str,
		run_started_at: datetime,
		usage_events: list[dict[str, Any]],
	) -> None:
		provider_model_breakdown: dict[str, dict[str, Any]] = {}
		totals = {
			"input_tokens": 0,
			"input_image_tokens": 0,
			"input_total_tokens": 0,
			"thoughts_tokens": 0,
			"cached_content_tokens": 0,
			"tool_use_prompt_tokens": 0,
			"output_tokens": 0,
			"accounted_total_tokens": 0,
			"unaccounted_tokens": 0,
			"total_tokens": 0,
		}

		for event in usage_events:
			for key in totals:
				totals[key] += int(event.get(key) or 0)

			provider = str(event.get("provider") or "unknown")
			model = str(event.get("model") or "unknown")
			bucket_key = f"{provider}::{model}"
			bucket = provider_model_breakdown.setdefault(
				bucket_key,
				{
					"provider": provider,
					"model": model,
					"calls": 0,
					"total_tokens": 0,
					"input_total_tokens": 0,
					"output_tokens": 0,
				},
			)
			bucket["calls"] += 1
			bucket["total_tokens"] += int(event.get("total_tokens") or 0)
			bucket["input_total_tokens"] += int(event.get("input_total_tokens") or 0)
			bucket["output_tokens"] += int(event.get("output_tokens") or 0)

		calls = len(usage_events)
		denominator = calls if calls > 0 else 1
		averages = {
			"input_total_tokens": round(totals["input_total_tokens"] / denominator, 2),
			"output_tokens": round(totals["output_tokens"] / denominator, 2),
			"total_tokens": round(totals["total_tokens"] / denominator, 2),
		}

		structured_payload = {
			"metadata": {
				"run_id": run_id,
				"started_at": run_started_at.isoformat(),
				"generated_at": datetime.now(timezone.utc).isoformat(),
				"source_file": str(source_jsonl_file),
				"provider": self.provider_name,
				"calls": calls,
			},
			"summary": {
				"totals": totals,
				"averages_per_call": averages,
				"provider_model_breakdown": list(provider_model_breakdown.values()),
			},
			"events": usage_events,
		}

		target_path = source_jsonl_file.with_suffix(".json")
		target_path.write_text(json.dumps(structured_payload, ensure_ascii=False, indent=2), encoding="utf-8")

	def _update_usage_totals(
		self,
		totals_file: Path,
		run_id: str,
		run_started_at: datetime,
		calls: int,
		input_tokens: int,
		input_image_tokens: int,
		input_total_tokens: int,
		thoughts_tokens: int,
		cached_content_tokens: int,
		tool_use_prompt_tokens: int,
		output_tokens: int,
		accounted_total_tokens: int,
		unaccounted_tokens: int,
		total_tokens: int,
	) -> None:
		if totals_file.exists():
			totals = json.loads(totals_file.read_text(encoding="utf-8"))
			if not isinstance(totals, dict):
				totals = {}
		else:
			totals = {}

		totals.setdefault("calls", 0)
		totals.setdefault("input_tokens", 0)
		totals.setdefault("input_image_tokens", 0)
		totals.setdefault("input_total_tokens", 0)
		totals.setdefault("thoughts_tokens", 0)
		totals.setdefault("cached_content_tokens", 0)
		totals.setdefault("tool_use_prompt_tokens", 0)
		totals.setdefault("output_tokens", 0)
		totals.setdefault("accounted_total_tokens", 0)
		totals.setdefault("unaccounted_tokens", 0)
		totals.setdefault("total_tokens", 0)
		totals.setdefault("runs", [])

		totals["calls"] += calls
		totals["input_tokens"] += input_tokens
		totals["input_image_tokens"] += input_image_tokens
		totals["input_total_tokens"] += input_total_tokens
		totals["thoughts_tokens"] += thoughts_tokens
		totals["cached_content_tokens"] += cached_content_tokens
		totals["tool_use_prompt_tokens"] += tool_use_prompt_tokens
		totals["output_tokens"] += output_tokens
		totals["accounted_total_tokens"] += accounted_total_tokens
		totals["unaccounted_tokens"] += unaccounted_tokens
		totals["total_tokens"] += total_tokens
		totals["runs"].append(
			{
				"run_id": run_id,
				"started_at": run_started_at.isoformat(),
				"provider": self.provider_name,
				"calls": calls,
				"input_tokens": input_tokens,
				"input_image_tokens": input_image_tokens,
				"input_total_tokens": input_total_tokens,
				"thoughts_tokens": thoughts_tokens,
				"cached_content_tokens": cached_content_tokens,
				"tool_use_prompt_tokens": tool_use_prompt_tokens,
				"output_tokens": output_tokens,
				"accounted_total_tokens": accounted_total_tokens,
				"unaccounted_tokens": unaccounted_tokens,
				"total_tokens": total_tokens,
			}
		)

		totals_file.write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")

	def _load_yaml(self, path: Path) -> dict[str, Any]:
		if not path.exists():
			return {}
		payload = yaml.safe_load(path.read_text(encoding="utf-8"))
		if not isinstance(payload, dict):
			return {}
		return payload

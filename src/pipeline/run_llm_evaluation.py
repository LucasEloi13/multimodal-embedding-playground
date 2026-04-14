"""Run LLM evaluation over query/top1 pairs."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from embedding.services.llm_evaluation import LLMEvaluationService
except ModuleNotFoundError:
    from src.embedding.services.llm_evaluation import LLMEvaluationService


logger = logging.getLogger(__name__)


def _configure_external_noise() -> None:
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "google_genai",
        "google_genai.models",
        "google",
        "google.auth",
        "google.api_core",
        "google.genai",
        "google.genai.models",
        "google.generativeai",
        "grpc",
    ]
    for name in noisy_loggers:
        ext_logger = logging.getLogger(name)
        ext_logger.setLevel(logging.WARNING)
        ext_logger.propagate = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM evaluation for query vs top1 candidates")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. INFO, WARNING)",
    )
    parser.add_argument(
        "--production-config",
        type=Path,
        default=PROJECT_ROOT / "src" / "infra" / "config" / "production.yml",
        help="Runtime config file",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "top1_joined" / "mean.top1_joined.json",
        help="Input query/top1 JSON file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output evaluation file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of pairs for smoke test",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional number of parallel workers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    _configure_external_noise()

    output_file = args.output_file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROJECT_ROOT / "data" / "debug" / f"llm_evaluation.{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    service = LLMEvaluationService(production_config_path=args.production_config)
    logger.info(
        "LLM evaluation started | provider=%s model=%s workers=%s limit=%s input=%s",
        service.provider_name,
        service.provider_cfg.get("model"),
        args.workers if args.workers is not None else service.default_workers,
        args.limit,
        args.input_file,
    )
    result = service.evaluate_file(
        input_path=args.input_file,
        output_path=output_file,
        limit=args.limit,
        workers=args.workers,
    )

    metadata = result.get("metadata", {})
    usage = metadata.get("usage", {})
    logger.info(
        "LLM evaluation finished | received=%s deduped=%s evaluated=%s total_tokens=%s output=%s",
        metadata.get("items_received"),
        metadata.get("items_after_dedup"),
        metadata.get("items_evaluated"),
        usage.get("total_tokens"),
        output_file,
    )
    print(
        f"LLM evaluation done: items={result.get('metadata', {}).get('items_evaluated')} "
        f"total_tokens={usage.get('total_tokens')} output={output_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
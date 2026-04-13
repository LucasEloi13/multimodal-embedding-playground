"""Run retrieval evaluation comparison for multiple rerank strategies using the same sample."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from core.utils.scrapped_listing_sampler import create_random_sample_file
except ModuleNotFoundError:
    from src.core.utils.scrapped_listing_sampler import create_random_sample_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation for multiple rerank strategies with a shared sample"
    )
    parser.add_argument(
        "--production-config",
        type=Path,
        default=PROJECT_ROOT / "src" / "infra" / "config" / "production.yml",
        help="Runtime config file",
    )
    parser.add_argument(
        "--embedding-config",
        type=Path,
        default=PROJECT_ROOT / "src" / "infra" / "config" / "embedding_models.yml",
        help="Embedding model config file",
    )
    parser.add_argument(
        "--scrapped-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "scrapped_listings",
        help="Directory with scrapped listing json files",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size override for generated shared sample",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="Optional existing sample file. If omitted, one sample is generated and reused.",
    )
    parser.add_argument(
        "--sample-output-file",
        type=Path,
        default=None,
        help="Optional path for generated sample query file",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["image_only", "mean", "text_only"],
        choices=["mean", "text_only", "image_only"],
        help="Rerank strategies to compare",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "debug",
        help="Directory for comparison outputs",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="retrieval_evaluation_compare",
        help="Prefix for generated output filenames",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank-k", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-samples-per-output-file", type=int, default=None)
    parser.add_argument("--text-index-path", type=Path, default=None)
    parser.add_argument("--image-index-path", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _build_output_path(output_dir: Path, output_prefix: str, timestamp: str, strategy: str) -> Path:
    filename = f"{output_prefix}.{timestamp}.{strategy}.json"
    return output_dir / filename


def _run_single_strategy(args: argparse.Namespace, query_file: Path, strategy: str, output_file: Path) -> None:
    command: list[str] = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "pipeline" / "run_retrieval_evaluation.py"),
        "--production-config",
        str(args.production_config),
        "--embedding-config",
        str(args.embedding_config),
        "--scrapped-dir",
        str(args.scrapped_dir),
        "--query-file",
        str(query_file),
        "--rerank-strategy",
        strategy,
        "--output-file",
        str(output_file),
        "--log-level",
        args.log_level,
    ]

    optional_flags: list[tuple[str, object | None]] = [
        ("--top-k", args.top_k),
        ("--rerank-k", args.rerank_k),
        ("--workers", args.workers),
        ("--max-samples-per-output-file", args.max_samples_per_output_file),
        ("--text-index-path", args.text_index_path),
        ("--image-index-path", args.image_index_path),
    ]
    for flag, value in optional_flags:
        if value is not None:
            command.extend([flag, str(value)])

    print(f"Running strategy={strategy} -> {output_file}")
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.query_file is not None:
        query_file = args.query_file
    else:
        sample_size = args.sample_size if args.sample_size is not None else 100
        query_file, sample = create_random_sample_file(
            scrapped_dir=args.scrapped_dir,
            sample_size=sample_size,
            output_file=args.sample_output_file,
        )
        print(f"Shared sample generated: {query_file} ({len(sample)} items)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Comparing strategies using shared sample: {query_file}")

    for strategy in args.strategies:
        output_file = _build_output_path(
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            timestamp=timestamp,
            strategy=strategy,
        )
        _run_single_strategy(args=args, query_file=query_file, strategy=strategy, output_file=output_file)

    print("Comparison completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Utilities to build random query samples from scrapped listing JSON files."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from core.utils.reference_product_loader import JsonReferenceProductParser
except ModuleNotFoundError:
    from src.core.utils.reference_product_loader import JsonReferenceProductParser


def collect_scrapped_listings(scrapped_dir: str | Path) -> list[dict[str, Any]]:
    source_dir = Path(scrapped_dir)
    parser = JsonReferenceProductParser()

    listings: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.json")):
        try:
            payload = path.read_text(encoding="utf-8")
            records = parser.parse(payload)
            listings.extend(record for record in records if isinstance(record, dict))
        except Exception:
            continue

    return listings


def generate_random_sample(
    listings: list[dict[str, Any]],
    sample_size: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []

    if seed is not None:
        random.seed(seed)

    sample_size = min(sample_size, len(listings))
    if sample_size == len(listings):
        return list(listings)

    return random.sample(listings, sample_size)


def write_sample_jsonl(
    sample: list[dict[str, Any]],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for item in sample:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_path


def create_random_sample_file(
    scrapped_dir: str | Path,
    sample_size: int,
    output_file: str | Path | None = None,
    seed: int | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    listings = collect_scrapped_listings(scrapped_dir)
    sample = generate_random_sample(listings, sample_size=sample_size, seed=seed)

    target = Path(output_file) if output_file is not None else _default_output_file()
    written = write_sample_jsonl(sample, target)
    return written, sample


def _default_output_file() -> Path:
    root = Path(__file__).resolve().parents[3]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "data" / "debug" / "samples" / f"scrapped_sample.{timestamp}.jsonl"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Build random sample from scrapped listings")
    parser.add_argument(
        "--scrapped-dir",
        type=Path,
        default=root / "data" / "scrapped_listings",
        help="Directory containing scrapped JSON files",
    )
    parser.add_argument("--sample-size", type=int, required=True, help="Random sample size")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output sample file path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file, sample = create_random_sample_file(
        scrapped_dir=args.scrapped_dir,
        sample_size=args.sample_size,
        output_file=args.output_file,
        seed=args.seed,
    )

    print(
        json.dumps(
            {
                "scrapped_dir": str(args.scrapped_dir),
                "sample_size": len(sample),
                "output_file": str(output_file),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

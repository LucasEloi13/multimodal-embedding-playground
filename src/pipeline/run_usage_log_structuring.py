"""Convert usage_*.jsonl logs into structured JSON files for easier inspection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TOKEN_FIELDS = [
    "input_tokens",
    "input_image_tokens",
    "input_total_tokens",
    "thoughts_tokens",
    "cached_content_tokens",
    "tool_use_prompt_tokens",
    "output_tokens",
    "accounted_total_tokens",
    "unaccounted_tokens",
    "total_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert usage_*.jsonl to structured JSON")
    parser.add_argument(
        "--usage-dir",
        type=Path,
        default=PROJECT_ROOT / "src" / "infra" / "llm" / "usage_logs",
        help="Directory containing usage_*.jsonl files",
    )
    parser.add_argument(
        "--pattern",
        default="usage_*.jsonl",
        help="Glob pattern used to find usage log files",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single usage_*.jsonl file to convert",
    )
    return parser.parse_args()


def load_usage_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.strip()
        if not content:
            continue
        events.append(json.loads(content))
    return events


def build_structured_payload(source_file: Path, usage_events: list[dict[str, Any]]) -> dict[str, Any]:
    provider_model_breakdown: dict[str, dict[str, Any]] = {}
    totals = {key: 0 for key in TOKEN_FIELDS}

    for event in usage_events:
        for key in TOKEN_FIELDS:
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
    averages_per_call = {
        "input_total_tokens": round(totals["input_total_tokens"] / denominator, 2),
        "output_tokens": round(totals["output_tokens"] / denominator, 2),
        "total_tokens": round(totals["total_tokens"] / denominator, 2),
    }

    provider = str(usage_events[0].get("provider") or "unknown") if usage_events else "unknown"
    run_id = source_file.stem.replace("usage_", "", 1)

    return {
        "metadata": {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source_file),
            "provider": provider,
            "calls": calls,
        },
        "summary": {
            "totals": totals,
            "averages_per_call": averages_per_call,
            "provider_model_breakdown": list(provider_model_breakdown.values()),
        },
        "events": usage_events,
    }


def convert_file(path: Path) -> Path:
    usage_events = load_usage_events(path)
    structured_payload = build_structured_payload(path, usage_events)
    output_path = path.with_suffix(".json")
    output_path.write_text(json.dumps(structured_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()

    if args.input_file is not None:
        files = [args.input_file]
    else:
        files = sorted(args.usage_dir.glob(args.pattern))

    if not files:
        print("No files found for conversion")
        return 0

    converted = 0
    for path in files:
        if not path.name.startswith("usage_"):
            continue
        convert_file(path)
        converted += 1

    print(f"Converted {converted} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

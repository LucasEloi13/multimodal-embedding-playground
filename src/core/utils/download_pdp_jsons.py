"""Download PDP JSON files from S3 using AWS CLI authentication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import threading
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "scrapped_listings"
DEFAULT_REQUESTS_PER_SECOND = 10
DEFAULT_WORKERS = 10
PROGRESS_BAR_WIDTH = 32


class _RateLimiter:
	def __init__(self, requests_per_second: int) -> None:
		self._min_interval = 1.0 / max(1, requests_per_second)
		self._next_allowed = 0.0
		self._lock = threading.Lock()

	def wait_turn(self) -> None:
		with self._lock:
			now = time.monotonic()
			if now < self._next_allowed:
				sleep_for = self._next_allowed - now
				self._next_allowed += self._min_interval
			else:
				sleep_for = 0.0
				self._next_allowed = now + self._min_interval

		if sleep_for > 0:
			time.sleep(sleep_for)


def _normalize_prefix(prefix: str) -> str:
	cleaned = prefix.strip().lstrip("/")
	if cleaned and not cleaned.endswith("/"):
		cleaned += "/"
	return cleaned


def _render_progress(downloaded: int, total: int) -> str:
	if total <= 0:
		return "[--------------------------------] 0.0% (0/0)"

	ratio = downloaded / total
	filled = int(PROGRESS_BAR_WIDTH * ratio)
	bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
	return f"[{bar}] {ratio * 100:5.1f}% ({downloaded}/{total})"


def _download_all(bucket_name: str, prefix: str, output_dir: Path) -> tuple[int, int]:
	s3 = boto3.client("s3")
	paginator = s3.get_paginator("list_objects_v2")
	limiter = _RateLimiter(requests_per_second=DEFAULT_REQUESTS_PER_SECOND)

	keys: list[str] = []

	for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
		for obj in page.get("Contents", []):
			key = obj.get("Key")
			if not key or key.endswith("/"):
				continue
			keys.append(key)

	print(f"Arquivos detectados: {len(keys)}")
	if keys:
		print(_render_progress(0, len(keys)), end="\r", flush=True)

	def _download_one(key: str) -> bool:
		limiter.wait_turn()
		relative = key[len(prefix) :] if key.startswith(prefix) else Path(key).name
		target = output_dir / relative
		target.parent.mkdir(parents=True, exist_ok=True)

		s3.download_file(bucket_name, key, str(target))
		return True

	downloaded = 0
	with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
		futures = [executor.submit(_download_one, key) for key in keys]
		for future in as_completed(futures):
			downloaded += int(future.result())
			print(_render_progress(downloaded, len(keys)), end="\r", flush=True)

	if keys:
		print(_render_progress(downloaded, len(keys)))

	return len(keys), downloaded


def main() -> int:
	load_dotenv()

	bucket_name = os.getenv("BUCKET_NAME")
	folder_path = os.getenv("BUCKET_FOLDER_PATH")

	if not bucket_name:
		print("Missing BUCKET_NAME in environment/.env", file=sys.stderr)
		return 1

	if not folder_path:
		print("Missing BUCKET_FOLDER_PATH in environment/.env", file=sys.stderr)
		return 1

	prefix = _normalize_prefix(folder_path)
	output_dir = DEFAULT_OUTPUT_DIR
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		listed, downloaded = _download_all(
			bucket_name=bucket_name,
			prefix=prefix,
			output_dir=output_dir,
		)
	except (BotoCoreError, ClientError) as exc:
		print(f"S3 error: {exc}", file=sys.stderr)
		return 2

	if listed == 0:
		print("No files found under the provided prefix.")
		return 0

	print(f"Done. Files listed: {listed}, downloaded: {downloaded}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

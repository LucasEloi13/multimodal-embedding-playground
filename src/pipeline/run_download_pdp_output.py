"""Pipeline entrypoint that delegates to core download utility."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from core.utils.download_pdp_jsons import main
except ModuleNotFoundError:
    from src.core.utils.download_pdp_jsons import main


if __name__ == "__main__":
    raise SystemExit(main())

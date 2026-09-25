from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    d = os.environ.get("PRIORS_CACHE_DIR")
    return Path(d) if d else Path.home() / ".priors" / "cache"

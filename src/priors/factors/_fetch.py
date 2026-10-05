"""Download-and-cache helper for public data files.

Every public source goes through ``fetch``. The raw bytes are cached under
``<cache_dir>/raw/<key>``. A cached copy is reused until it is older than
``max_age_days``. If a refresh fails (these sites are occasionally unreachable),
the stale copy is used and a warning is emitted, so a flaky network never stops
a class session.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import requests

from ..data.cache import cache_dir

USER_AGENT = "Mozilla/5.0 (priors-quant; academic use)"


class FetchError(RuntimeError):
    pass


def raw_dir() -> Path:
    d = cache_dir() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, retries: int, timeout: int) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            if not r.content:
                raise FetchError(f"empty response from {url}")
            return r.content
        except Exception as e:  # network errors are routine here; retry all of them
            last = e
            time.sleep(min(2 * (attempt + 1), 10))
    raise FetchError(f"could not download {url} after {retries} attempts: {last}")


def fetch(
    url: str,
    key: str,
    max_age_days: float = 30,
    retries: int = 6,
    timeout: int = 120,
    refresh: bool = False,
) -> bytes:
    """Return the bytes at ``url``, using the cache file ``key`` when it is fresh."""
    path = raw_dir() / key
    if path.exists() and not refresh:
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return path.read_bytes()
    try:
        content = _download(url, retries, timeout)
    except FetchError:
        if path.exists():
            warnings.warn(f"Using a stale cached copy of {key}: refresh failed.", stacklevel=2)
            return path.read_bytes()
        raise
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)
    return content


def cached_at(key: str) -> float | None:
    """Modification time of a cached file (used as the data snapshot stamp)."""
    path = raw_dir() / key
    return path.stat().st_mtime if path.exists() else None

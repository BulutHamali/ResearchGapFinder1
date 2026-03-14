import hashlib
import logging
import os
from typing import Any, Callable, Optional

import diskcache

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 86400  # 24 hours


class Cache:
    """Disk-based cache wrapping diskcache.Cache."""

    def __init__(self, cache_dir: str = ".cache", size_limit: int = 2 ** 30):
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = diskcache.Cache(cache_dir, size_limit=size_limit)
        logger.info(f"Cache initialized at '{cache_dir}' (size_limit={size_limit // 1024 // 1024} MB)")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache. Returns None if not found or expired."""
        value = self._cache.get(key, default=None)
        if value is not None:
            logger.debug(f"Cache HIT: {key[:80]}")
        else:
            logger.debug(f"Cache MISS: {key[:80]}")
        return value

    def set(self, key: str, value: Any, ttl: int = _DEFAULT_TTL) -> None:
        """Store a value in the cache with a TTL (seconds)."""
        self._cache.set(key, value, expire=ttl)
        logger.debug(f"Cache SET: {key[:80]} (ttl={ttl}s)")

    def get_or_compute(self, key: str, func: Callable, ttl: int = _DEFAULT_TTL) -> Any:
        """Return cached value or compute it via func(), cache and return the result."""
        value = self.get(key)
        if value is not None:
            return value
        value = func()
        self.set(key, value, ttl=ttl)
        return value

    def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        self._cache.delete(key)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()

    def close(self) -> None:
        """Close the underlying cache connection."""
        self._cache.close()

    @staticmethod
    def make_key(*args: Any) -> str:
        """Build a stable cache key from arbitrary arguments."""
        raw = str(args)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Module-level default cache instance (lazy-initialized)
_default_cache: Optional[Cache] = None


def get_cache(cache_dir: str = ".cache") -> Cache:
    """Return (or create) the default module-level cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache(cache_dir=cache_dir)
    return _default_cache

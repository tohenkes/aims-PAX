"""A caching mechanism for Model attributes."""

from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class CacheEntry:
    value: Any
    valid: bool = True

    def invalidate(self):
        self.valid = False


class Cache:
    """A cache with invalidation, stored as subdicts."""

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None or not entry.valid:
            return None
        return entry.value

    def set(self, key: str, value: Any):
        self._store[key] = CacheEntry(value)

    def invalidate(self, key: str = None):
        """Invalidate a specific key, a whole namespace, or everything."""
        if key is None:
            for entry in self._store.values():
                entry.invalidate()
        else:
            entry = self._store.get(key)
            if entry:
                entry.invalidate()

    def cached(self, key: str, fn: Callable) -> Any:
        """Get from cache or compute and store."""
        result = self.get(key)
        if result is None:
            result = fn()
            self.set(key, result)
        return result
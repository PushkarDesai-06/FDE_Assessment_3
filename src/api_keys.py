"""Round-robin API key manager.

Keys come from the environment:
    GROQ_API_KEYS=key1,key2,key3     (comma-separated, preferred)
    GROQ_API_KEY / LLM_API_KEY       (single key, also added to the pool)

Every LLM request takes the next key in rotation. A key that hits a rate limit
(or is rejected) is rested for a cooldown and skipped until it expires; the
request moves straight on to the next key instead of sleeping. Only when every
key is resting does the caller wait, for the shortest remaining cooldown.

Keys are never logged; telemetry refers to them as "key#1", "key#2", ...
"""
from __future__ import annotations

import os
import threading
import time

from openai import OpenAI

INVALID_KEY_COOLDOWN_S = 3600.0


class NoApiKeysError(RuntimeError):
    pass


def keys_from_env() -> list[str]:
    raw = [os.getenv("GROQ_API_KEYS", ""), os.getenv("GROQ_API_KEY", ""), os.getenv("LLM_API_KEY", "")]
    keys: list[str] = []
    for value in raw:
        for key in value.split(","):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


class ApiKeyManager:
    def __init__(self, keys: list[str]):
        if not keys:
            raise NoApiKeysError("No GROQ_API_KEYS / GROQ_API_KEY / LLM_API_KEY configured")
        self._keys = list(keys)
        self._next = 0
        self._cooldown_until = {k: 0.0 for k in self._keys}
        self._clients: dict[str, OpenAI] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    def label(self, key: str) -> str:
        return f"key#{self._keys.index(key) + 1}"

    def acquire(self) -> tuple[str, float]:
        """Return (key, seconds_to_wait). Wait is 0 unless every key is cooling down."""
        with self._lock:
            now = time.monotonic()
            n = len(self._keys)
            for offset in range(n):
                key = self._keys[(self._next + offset) % n]
                if self._cooldown_until[key] <= now:
                    self._next = (self._keys.index(key) + 1) % n
                    return key, 0.0
            key = min(self._keys, key=self._cooldown_until.__getitem__)
            self._next = (self._keys.index(key) + 1) % n
            return key, self._cooldown_until[key] - now

    def penalize(self, key: str, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[key] = max(self._cooldown_until[key], time.monotonic() + seconds)

    def client(self, key: str) -> OpenAI:
        with self._lock:
            if key not in self._clients:
                self._clients[key] = OpenAI(
                    api_key=key,
                    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                    max_retries=0,
                    timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
                )
            return self._clients[key]


_manager: ApiKeyManager | None = None
_manager_keys: tuple[str, ...] = ()
_manager_lock = threading.Lock()


def get_key_manager() -> ApiKeyManager:
    """Process-wide manager, so rotation continues across requests. Rebuilt if the env keys change."""
    global _manager, _manager_keys
    keys = tuple(keys_from_env())
    with _manager_lock:
        if _manager is None or keys != _manager_keys:
            _manager, _manager_keys = ApiKeyManager(list(keys)), keys
        return _manager

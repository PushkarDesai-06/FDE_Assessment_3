"""Round-robin API key manager tests. No network calls."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import openai

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import api_keys  # noqa: E402
from src.api_keys import ApiKeyManager, NoApiKeysError, keys_from_env  # noqa: E402
from src.llm import LLM  # noqa: E402
from src.tools import Telemetry  # noqa: E402


class KeyManagerTests(unittest.TestCase):
    def test_round_robin(self):
        m = ApiKeyManager(["a", "b", "c"])
        self.assertEqual([m.acquire()[0] for _ in range(5)], ["a", "b", "c", "a", "b"])

    def test_cooling_key_is_skipped(self):
        m = ApiKeyManager(["a", "b", "c"])
        m.penalize("b", 60)
        self.assertEqual([m.acquire()[0] for _ in range(4)], ["a", "c", "a", "c"])

    def test_all_cooling_returns_shortest_wait(self):
        m = ApiKeyManager(["a", "b"])
        m.penalize("a", 30)
        m.penalize("b", 5)
        key, wait = m.acquire()
        self.assertEqual(key, "b")
        self.assertTrue(0 < wait <= 5)

    def test_env_parsing_dedupes_and_merges(self):
        env = {"GROQ_API_KEYS": " k1, k2 ,,k1", "GROQ_API_KEY": "k3", "LLM_API_KEY": "k2"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(keys_from_env(), ["k1", "k2", "k3"])

    def test_no_keys(self):
        with self.assertRaises(NoApiKeysError):
            ApiKeyManager([])

    def test_labels_never_expose_keys(self):
        m = ApiKeyManager(["secret-1", "secret-2"])
        self.assertEqual(m.label("secret-2"), "key#2")


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={"retry-after": "30"})
    return openai.RateLimitError("rate limited", response=response, body=None)


class LLMRotationTests(unittest.TestCase):
    def test_rate_limited_key_rotates_without_sleeping(self):
        used: list[str] = []
        ok = SimpleNamespace(choices=[SimpleNamespace(message="hello")], usage=None)

        def fake_client(key):
            def create(**_kwargs):
                used.append(key)
                if key == "k1":
                    raise _rate_limit_error()
                return ok
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        manager = ApiKeyManager(["k1", "k2"])
        manager.client = fake_client
        telemetry = Telemetry()
        with mock.patch.object(api_keys, "get_key_manager", return_value=manager), \
                mock.patch("src.llm.get_key_manager", return_value=manager), \
                mock.patch("src.llm.time.sleep") as sleep:
            msg = LLM(telemetry).chat("test", [], [], "required")
        self.assertEqual(msg, "hello")
        self.assertEqual(used, ["k1", "k2"])
        sleep.assert_not_called()
        self.assertEqual(telemetry.rate_limit_wait_s, 0)
        self.assertIn("key#1", telemetry.last_rate_limit)
        self.assertNotIn("k1", telemetry.last_rate_limit.replace("key#1", ""))


if __name__ == "__main__":
    unittest.main()

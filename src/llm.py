"""Thin chat-completions client (OpenAI SDK against Groq's OpenAI-compatible API).

Each request uses the next API key from the round-robin pool (src/api_keys.py).
Rate limits and transient errors are handled here so telemetry can separate real
model latency from time spent waiting on provider rate limits.
"""
from __future__ import annotations

import os
import time

import openai
from src.api_keys import INVALID_KEY_COOLDOWN_S, NoApiKeysError, get_key_manager
from src.tools import Telemetry


class LLMError(RuntimeError):
    pass


class LLMOutputError(LLMError):
    """The model kept producing output the provider could not parse as a tool call."""


class LLM:
    def __init__(self, telemetry: Telemetry):
        try:
            self.keys = get_key_manager()
        except NoApiKeysError as exc:
            raise LLMError(str(exc)) from exc
        self.model = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
        self.reasoning_effort = os.getenv("REASONING_EFFORT", "low")
        self.telemetry = telemetry

    def chat(self, stage: str, messages: list[dict], tools: list[dict], tool_choice) -> object:
        transient_failures = 0
        for _ in range(8 + 2 * len(self.keys)):
            key, wait = self.keys.acquire()
            if wait > 0:  # every key is cooling down
                wait = min(wait, 60.0)
                self.telemetry.rate_limit_wait_s += wait
                time.sleep(wait)
            start = time.perf_counter()
            try:
                resp = self.keys.client(key).chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    reasoning_effort=self.reasoning_effort,
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
                    max_completion_tokens=4096,
                )
            except openai.RateLimitError as exc:
                # Rest this key and rotate to the next one immediately.
                self.keys.penalize(key, _retry_after(exc) or 5.0)
                self.telemetry.last_rate_limit = f"{self.keys.label(key)}: {str(exc)[:300]}"
                continue
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                self.keys.penalize(key, INVALID_KEY_COOLDOWN_S)
                self.telemetry.last_rate_limit = f"{self.keys.label(key)} rejected: {type(exc).__name__}"
                continue
            except openai.BadRequestError as exc:
                # Groq returns 400 'tool_use_failed' / 'output_parse_failed' when the model emits malformed output.
                if any(code in str(exc) for code in ("tool_use_failed", "output_parse_failed")) and transient_failures < 2:
                    transient_failures += 1
                    self.telemetry.record_llm(stage, time.perf_counter() - start, note="malformed tool call; retried")
                    continue
                if any(code in str(exc) for code in ("tool_use_failed", "output_parse_failed")):
                    raise LLMOutputError(f"{stage}: {str(exc)[:300]}") from exc
                raise LLMError(f"{stage}: {exc}") from exc
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                transient_failures += 1
                if transient_failures > 2:
                    raise LLMError(f"{stage}: {type(exc).__name__}: {exc}") from exc
                time.sleep(2 ** transient_failures)
                continue
            except openai.APIError as exc:
                raise LLMError(f"{stage}: {type(exc).__name__}: {exc}") from exc
            self.telemetry.record_llm(stage, time.perf_counter() - start, usage=resp.usage, note=self.keys.label(key))
            return resp.choices[0].message
        raise LLMError(f"{stage}: gave up after repeated rate limits / errors")


def _retry_after(exc: openai.RateLimitError) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) + 0.5 if value else None
    except Exception:
        return None


def assistant_message(msg) -> dict:
    """Re-serialise an assistant turn (drop provider-specific fields like 'reasoning')."""
    out: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return out

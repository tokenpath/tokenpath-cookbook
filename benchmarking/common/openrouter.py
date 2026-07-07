"""OpenRouter chat client — the one LLM path for generation, the prompted
citation baseline, and the judge.

OpenRouter is OpenAI-compatible. We enable its usage-accounting extension
(`usage: {include: true}`) so every response carries the *actual* dollar cost of
the call, which feeds the $/query column directly instead of a price table that
drifts. Latency is measured at the call site.

Retries: exponential backoff on 429 / 5xx / transient network errors, capped.
Deterministic where it matters — temperature defaults to 0 for the judge and the
frozen-answer generator so a re-run reproduces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from .. import config


class OpenRouterError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float  # actual, from OpenRouter usage accounting (0.0 if absent)
    seconds: float
    model: str
    raw: dict = field(repr=False, default_factory=dict)


class OpenRouterClient:
    def __init__(self, api_key: str, api_url: str | None = None, timeout: int = 180):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required (https://openrouter.ai/keys)")
        self.api_key = api_key
        self.api_url = (api_url or config.OPENROUTER_API_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: str | list[str] | None = None,
        max_retries: int = 5,
    ) -> ChatResult:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},  # ask OpenRouter to report real cost
        }
        if stop is not None:
            payload["stop"] = stop

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                resp = self._session.post(
                    f"{self.api_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        # Optional attribution headers OpenRouter recommends:
                        "HTTP-Referer": "https://github.com/tokenpath/tokenpath-cookbook",
                        "X-Title": "TokenPath benchmark",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                seconds = time.perf_counter() - t0
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise OpenRouterError(f"retryable {resp.status_code}: {resp.text[:200]}")
                if not resp.ok:
                    raise OpenRouterError(f"{resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                choice = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {}) or {}
                return ChatResult(
                    text=choice,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    cost_usd=float(usage.get("cost", 0.0) or 0.0),
                    seconds=seconds,
                    model=data.get("model", model),
                    raw=data,
                )
            except (OpenRouterError, requests.RequestException) as exc:
                last_err = exc
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)  # 1, 2, 4, 8s
        raise OpenRouterError(f"chat failed after {max_retries} attempts: {last_err}")

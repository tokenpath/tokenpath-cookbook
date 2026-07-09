"""Multi-backend LLM chat client — the one LLM path for generation, the prompted
citation baseline, and the judge.

Every model is addressed by a provider-prefixed id and routed to that provider's
OpenAI-compatible endpoint:

    openai/<model>   -> https://api.openai.com/v1            (OPENAI_API_KEY)
    google/<model>   -> Gemini OpenAI-compat endpoint        (GOOGLE_API_KEY)
    <anything else>  -> OpenRouter                           (OPENROUTER_API_KEY)

We call each provider directly (not through OpenRouter) so a personal OpenRouter
balance isn't consumed and so the benchmark can pin exact frontier models. Only
the OpenRouter fallback reports real dollar cost via its usage-accounting
extension; for the direct providers we price tokens from config.PRICE (list
prices), which feeds the $/query column.

Provider quirks handled here so call sites stay uniform:
  - OpenAI reasoning models (gpt-5.x) reject `max_tokens` (need
    `max_completion_tokens`) and reject any non-default `temperature`; they also
    spend hidden reasoning tokens, so we floor the completion budget and request
    low reasoning effort for these generation/citation tasks.
  - Gemini 2.5 is a thinking model; with a small token budget it spends it all on
    hidden reasoning and returns empty content. We send `reasoning_effort:"none"`
    so the short-output judge gets its rating within budget.

Retries: exponential backoff on 429 / 5xx / transient network errors, capped.
Latency is measured at the call site.
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
    cost_usd: float  # real (OpenRouter usage accounting) or priced from config.PRICE
    seconds: float
    model: str
    raw: dict = field(repr=False, default_factory=dict)


# Provider backends. base_url is OpenAI-compatible /chat/completions for all three.
_BACKENDS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", "GOOGLE_API_KEY"),
    "openrouter": (config.OPENROUTER_API_URL, "OPENROUTER_API_KEY"),
}


def _route(model: str) -> tuple[str, str]:
    """(provider, model_name) — direct providers strip the prefix; else OpenRouter."""
    prefix = model.split("/", 1)[0]
    if prefix in ("openai", "google"):
        return prefix, model.split("/", 1)[1]
    return "openrouter", model  # OpenRouter keeps the fully-qualified id


class LLMClient:
    """Routes each chat call to the right provider by the model's prefix."""

    def __init__(self, keys: dict[str, str] | None = None, timeout: int = 180):
        import os

        # One API key per provider, read from the environment unless supplied.
        self.keys = keys or {
            provider: os.environ.get(envname, "")
            for provider, (_, envname) in _BACKENDS.items()
        }
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
        provider, model_name = _route(model)
        base_url, _ = _BACKENDS[provider]
        api_key = self.keys.get(provider, "")
        if not api_key:
            raise OpenRouterError(f"missing API key for provider '{provider}' (model {model})")

        payload: dict = {"model": model_name, "messages": messages}
        if stop is not None:
            payload["stop"] = stop

        if provider == "openai":
            # Reasoning models: max_completion_tokens, no temperature, low effort.
            payload["max_completion_tokens"] = max(max_tokens, 4096)
            payload["reasoning_effort"] = "low"
        elif provider == "google":
            # Disable Gemini thinking so short-budget judge calls return content.
            payload["max_tokens"] = max(max_tokens, 16)
            payload["temperature"] = temperature
            payload["reasoning_effort"] = "none"
        else:  # openrouter
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
            payload["usage"] = {"include": True}  # ask for real cost

        headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/tokenpath/tokenpath-cookbook"
            headers["X-Title"] = "TokenPath benchmark"

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                resp = self._session.post(
                    f"{base_url}/chat/completions", headers=headers,
                    json=payload, timeout=self.timeout,
                )
                seconds = time.perf_counter() - t0
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise OpenRouterError(f"retryable {resp.status_code}: {resp.text[:200]}")
                if not resp.ok:
                    raise OpenRouterError(f"{resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                text = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage", {}) or {}
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                cost = usage.get("cost")
                if cost is None:  # direct providers don't report cost -> price it
                    inp, out = config.PRICE.get(model, (0.0, 0.0))
                    cost = pt / 1_000_000 * inp + ct / 1_000_000 * out
                return ChatResult(
                    text=text, prompt_tokens=pt, completion_tokens=ct,
                    cost_usd=float(cost), seconds=seconds,
                    model=data.get("model", model), raw=data,
                )
            except (OpenRouterError, requests.RequestException) as exc:
                last_err = exc
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)  # 1, 2, 4, 8s
        raise OpenRouterError(f"chat failed after {max_retries} attempts: {last_err}")


# Backwards-compatible alias — call sites and type hints still import OpenRouterClient.
OpenRouterClient = LLMClient

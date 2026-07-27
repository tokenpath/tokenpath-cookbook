"""Client construction from the environment.

Secrets come from env vars only — never checked into the repo. Set them before
running (see benchmarking/README.md):

    export TOKENPATH_API_KEY=tpk_...
    export OPENROUTER_API_KEY=sk-or-...
    export ANTHROPIC_API_KEY=sk-ant-...   # optional; only for the Citations API baseline
"""

from __future__ import annotations

import os

from .openrouter import LLMClient
from .tokenpath import TokenPathClient


def tokenpath_client() -> TokenPathClient:
    return TokenPathClient(os.environ.get("TOKENPATH_API_KEY", ""))


def llm_client(timeout: int | None = None) -> LLMClient:
    """Multi-backend chat client; reads OPENAI/GOOGLE/OPENROUTER keys from env.

    The default 180s read timeout is fine for hosted providers but not for a local
    thinking model given a large completion budget: reasoning plus a long citation
    JSON can generate for many minutes, and a timeout there is scored as the model
    failing to cite. Exp 4's thinking row raises this (TP_LLM_TIMEOUT_S).
    """
    return LLMClient(timeout=timeout or int(os.environ.get("TP_LLM_TIMEOUT_S", "180")))


# Back-compat alias — call sites still say openrouter_client().
openrouter_client = llm_client


def anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None

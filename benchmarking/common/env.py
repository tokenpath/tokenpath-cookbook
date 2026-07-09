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


def llm_client() -> LLMClient:
    """Multi-backend chat client; reads OPENAI/GOOGLE/OPENROUTER keys from env."""
    return LLMClient()


# Back-compat alias — call sites still say openrouter_client().
openrouter_client = llm_client


def anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None

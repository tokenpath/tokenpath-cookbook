"""Pinned configuration for the TokenPath quality benchmark.

Everything a reviewer would need to reproduce our numbers lives here: model
versions, judge version, thresholds, dates, and pricing. Nothing that affects a
reported number should be hard-coded elsewhere.

Reproducibility bar (from the plan, non-negotiable):
  - Pin model versions, judge version, thresholds, dates.
  - Thresholds tuned on the VAL split, reported on the TEST split.
  - The judge is a single, consistent model across every row (including
    re-judged anchor rows) so the F1 column is apples-to-apples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Dates / provenance                                                          #
# --------------------------------------------------------------------------- #
# Stamp results with the date the run was frozen. We do NOT call datetime.now()
# inside the harness — the run date is passed in so a re-run reproduces byte for
# byte. Override with --run-date on the CLIs.
DEFAULT_RUN_DATE = "2026-07-07"

# --------------------------------------------------------------------------- #
# Endpoints                                                                   #
# --------------------------------------------------------------------------- #
TOKENPATH_API_URL = os.environ.get("TOKENPATH_API_URL", "https://api.tokenpath.ai")
OPENROUTER_API_URL = os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
# The Anthropic Citations API is a proprietary feature not exposed through
# OpenRouter's normalized chat API, so that one baseline talks to Anthropic
# directly. It is gated on ANTHROPIC_API_KEY and skipped if unset.
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com")

# --------------------------------------------------------------------------- #
# Models — pinned. Cheap by default (per instruction). Every model id is an    #
# OpenRouter slug except the Anthropic-native Citations baseline.              #
# --------------------------------------------------------------------------- #

# Models are provider-prefixed and routed to that provider's API directly
# (openai/* -> OpenAI, google/* -> Gemini, else -> OpenRouter). We use current
# frontier models — no retired models in a benchmark people are meant to trust.

# The single generator that produces the ONE frozen answer set (no citations).
# All post-hoc methods attribute these exact same answers.
GENERATOR_MODEL = os.environ.get("TP_GENERATOR_MODEL", "openai/gpt-5.5")

# The prompted-citation baseline ("add citations" pass over the frozen answer).
PROMPTED_MODEL = os.environ.get("TP_PROMPTED_MODEL", "openai/gpt-5.5")

# The judge. One consistent, cheap-but-modern judge across ALL rows (ours + any
# re-judged anchors) so the F1 column is internally apples-to-apples. Published
# anchors used gpt-4o-2024-05-13; we report those greyed for context.
JUDGE_MODEL = os.environ.get("TP_JUDGE_MODEL", "google/gemini-2.5-flash")

# The Anthropic Citations API reproduction-mode baseline (native Anthropic).
CITATIONS_API_MODEL = os.environ.get("TP_CITATIONS_MODEL", "claude-sonnet-5")

# Embedding-retrieval baseline — hosted OpenAI embeddings (text-embedding-3-large),
# the "just embed the answer and search the source" strawman with a current strong
# retriever. Reranking is off for now (a hosted reranker needs its own key; the
# Cohere trial is capped at 1000 calls/month, far below the ~14k this needs). A
# rerank stage can be layered back on later.
EMBED_MODEL = os.environ.get("TP_EMBED_MODEL", "text-embedding-3-large")
EMBED_USD_PER_MTOK = 0.13  # OpenAI text-embedding-3-large list price, for $/query
# Local cross-encoder reranker over the top-k retrieved sentences (modern, 2024;
# runs on CPU in ~16 min for the full set now that embedding is hosted). Set to
# "" / None for retrieval-only. Emits a logit we sigmoid to [0,1] relevance.
RERANK_MODEL = os.environ.get("TP_RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_SIGMOID = True

# --------------------------------------------------------------------------- #
# Pricing (USD). Used only for the $/query column. Update alongside model ids. #
# TokenPath: $1 per 1M attributed tokens (public pricing).                    #
# OpenRouter reports actual spend per call via usage accounting, so LLM $ come #
# from the API, not from this table — these are fallbacks only.               #
# --------------------------------------------------------------------------- #
TOKENPATH_USD_PER_MTOK = 1.0  # per 1M attributed tokens

# Per-model (input, output) USD per 1M tokens — list prices, used for the
# $/query column when calling providers directly (OpenAI / Gemini / Anthropic
# don't return a dollar cost the way OpenRouter's usage accounting does). Keep in
# sync with the pinned model ids above.
PRICE = {
    "openai/gpt-5.5": (5.0, 30.0),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "google/gemini-2.5-flash": (0.30, 2.50),
}
# Anthropic Citations baseline is priced separately (native, not via LLMClient).
ANTHROPIC_PRICE = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

# --------------------------------------------------------------------------- #
# Thresholds & method knobs                                                   #
# --------------------------------------------------------------------------- #

# TokenPath: a source SENTENCE is kept as a citation for a statement when its
# share of the statement's total attribution mass is at least this (mass is
# pooled per document sentence — LongBench-Cite's native citation unit). TUNED
# ON VAL, reported on TEST. The value checked in here is a placeholder
# overwritten by tune_threshold.py, which writes results/exp1_threshold.json.
# run.py reads that file if present.
TOKENPATH_MASS_THRESHOLD = 0.30

# Aggregation strategy for attention-mass -> sentence citations, tuned on the
# error dev set and validated on the full test split (F1 0.72 -> 0.785 reported /
# 0.75 -> 0.81 example-mean, closing ~70% of the gap to generation-time citation).
# Knobs are documented in common/aggregate.py.
#   row_norm       one-token-one-vote (stops a few high-magnitude tokens dominating)
#   threshold 0.30 higher mass bar -> drops junk citations -> precision + cleaner support
#   merge_adjacent fold adjacent supporting sentences into one passage -> recall
TOKENPATH_AGG = {"threshold": 0.30, "row_norm": True, "merge_adjacent": True, "max_spans": 4}

# Embedding baseline: retrieve top-k context sentences per statement, keep those
# whose rerank score clears the threshold (also tuned on val).
EMBED_TOP_K = 5  # candidates retrieved by cosine, then reranked
# Kept-citation bar: post-sigmoid rerank score when a reranker is set, else cosine.
EMBED_SCORE_THRESHOLD = 0.70  # post-sigmoid rerank bar (bge scores compress to ~0.5-0.73)

# Exp 2: single attribution-mass selection threshold applied identically to every
# search provider (no per-provider tuning — stated explicitly in the post).
WEBCODE_SELECTION_THRESHOLD = 0.15


@dataclass
class RunConfig:
    """Bundle passed through a run so every stage sees the same pins."""

    run_date: str = DEFAULT_RUN_DATE
    generator_model: str = GENERATOR_MODEL
    prompted_model: str = PROMPTED_MODEL
    judge_model: str = JUDGE_MODEL
    citations_api_model: str = CITATIONS_API_MODEL
    embed_model: str = EMBED_MODEL
    tokenpath_mass_threshold: float = TOKENPATH_MASS_THRESHOLD
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def as_provenance(self) -> dict:
        """The block we stamp into every results file."""
        return {
            "run_date": self.run_date,
            "generator_model": self.generator_model,
            "prompted_model": self.prompted_model,
            "judge_model": self.judge_model,
            "citations_api_model": self.citations_api_model,
            "embed_model": self.embed_model,
            "tokenpath_mass_threshold": self.tokenpath_mass_threshold,
            "seed": self.seed,
            "tokenpath_api_url": TOKENPATH_API_URL,
            "openrouter_api_url": OPENROUTER_API_URL,
        }

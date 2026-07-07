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

# The single generator that produces the ONE frozen answer set (no citations).
# All post-hoc methods attribute these exact same answers.
GENERATOR_MODEL = os.environ.get("TP_GENERATOR_MODEL", "openai/gpt-4o-mini")

# The prompted-citation baseline ("add citations" pass over the frozen answer).
PROMPTED_MODEL = os.environ.get("TP_PROMPTED_MODEL", "openai/gpt-4o-mini")

# The judge. LongBench-Cite's published rows used gpt-4o-2024-05-13. We run a
# single cheaper judge across ALL rows (ours + re-judged anchors) so the
# comparison is internally consistent. Flip this to "openai/gpt-4o-2024-05-13"
# for exact parity with the published anchor numbers.
JUDGE_MODEL = os.environ.get("TP_JUDGE_MODEL", "openai/gpt-4o-mini")

# The Anthropic Citations API reproduction-mode baseline (native Anthropic).
CITATIONS_API_MODEL = os.environ.get("TP_CITATIONS_MODEL", "claude-3-5-haiku-20241022")

# Embedding retrieve+rerank baseline (runs locally via sentence-transformers so
# it needs no third-party embedding key; downloads once from HuggingFace).
EMBED_MODEL = os.environ.get("TP_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.environ.get("TP_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --------------------------------------------------------------------------- #
# Pricing (USD). Used only for the $/query column. Update alongside model ids. #
# TokenPath: $1 per 1M attributed tokens (public pricing).                    #
# OpenRouter reports actual spend per call via usage accounting, so LLM $ come #
# from the API, not from this table — these are fallbacks only.               #
# --------------------------------------------------------------------------- #
TOKENPATH_USD_PER_MTOK = 1.0  # per 1M attributed tokens

# --------------------------------------------------------------------------- #
# Thresholds & method knobs                                                   #
# --------------------------------------------------------------------------- #

# TokenPath: a source span is kept as a citation for a statement when its
# attribution mass (fraction of the statement's total mass landing on that
# source region) is at least this. TUNED ON VAL, reported on TEST. The value
# checked in here is a placeholder overwritten by tune_threshold.py, which
# writes results/exp1_threshold.json. run.py reads that file if present.
TOKENPATH_MASS_THRESHOLD = 0.15

# Embedding baseline: retrieve top-k context sentences per statement, keep those
# whose rerank score clears the threshold (also tuned on val).
EMBED_TOP_K = 5
EMBED_SCORE_THRESHOLD = 0.0

# Exp 2: single attribution-mass threshold applied identically to every search
# provider (no per-provider tuning — stated explicitly in the post).
WEBCODE_MASS_THRESHOLD = 0.15


@dataclass
class RunConfig:
    """Bundle passed through a run so every stage sees the same pins."""

    run_date: str = DEFAULT_RUN_DATE
    generator_model: str = GENERATOR_MODEL
    prompted_model: str = PROMPTED_MODEL
    judge_model: str = JUDGE_MODEL
    citations_api_model: str = CITATIONS_API_MODEL
    embed_model: str = EMBED_MODEL
    rerank_model: str = RERANK_MODEL
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
            "rerank_model": self.rerank_model,
            "tokenpath_mass_threshold": self.tokenpath_mass_threshold,
            "seed": self.seed,
            "tokenpath_api_url": TOKENPATH_API_URL,
            "openrouter_api_url": OPENROUTER_API_URL,
        }

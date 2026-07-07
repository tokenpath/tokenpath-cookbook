"""TokenPath API client + attribution-mass helpers.

Two endpoints, mirroring the cookbook notebooks:

  POST /v1/attributions          span -> single best source span (+confidence)
  POST /v1/attributions/heatmap  raw sparse answer_token x document_token matrix

The heatmap is what the benchmark leans on: from it we can compute, for any
answer character span (a "statement"), how much attribution mass lands on each
region of the document. That mass distribution is what we threshold into
sentence-level citations, and its concentration is the confidence signal Exp 3
tests against ground truth.

Every call records wall-clock latency so the latency/$ column in the results
table comes from real measurements, never estimates. Timing is captured with
time.perf_counter at the call site (see common/timing.py for aggregation);
this module returns the elapsed seconds alongside each response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import requests

from .. import config


class TokenPathError(RuntimeError):
    """Raised on a non-2xx TokenPath response, carrying the request_id."""

    def __init__(self, status: int, body: str, request_id: str | None):
        self.status = status
        self.body = body
        self.request_id = request_id
        super().__init__(f"TokenPath {status} (request_id={request_id}): {body[:300]}")


@dataclass
class Timed:
    """A response paired with the wall-clock seconds it took."""

    value: dict
    seconds: float


class TokenPathClient:
    def __init__(self, api_key: str, api_url: str | None = None, timeout: int = 180):
        if not api_key:
            raise ValueError(
                "TOKENPATH_API_KEY is required — free key at https://platform.tokenpath.ai"
            )
        self.api_key = api_key
        self.api_url = (api_url or config.TOKENPATH_API_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    # Raw endpoints                                                       #
    # ------------------------------------------------------------------ #
    def _post(self, path: str, payload: dict) -> Timed:
        t0 = time.perf_counter()
        resp = self._session.post(
            f"{self.api_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        seconds = time.perf_counter() - t0
        if not resp.ok:
            raise TokenPathError(
                resp.status_code, resp.text, resp.headers.get("x-request-id")
            )
        return Timed(resp.json(), seconds)

    def attribute(
        self,
        document: str,
        question: str,
        answer: str,
        spans: Sequence[Sequence[int]],
        **options,
    ) -> Timed:
        """Resolve each answer char span to its single best source span."""
        return self._post(
            "/v1/attributions",
            {
                "document": document,
                "question": question,
                "answer": answer,
                "spans": list(spans),
                **options,
            },
        )

    def heatmap(self, document: str, question: str, answer: str, **options) -> Timed:
        """Return the raw sparse token-level attribution heatmap."""
        return self._post(
            "/v1/attributions/heatmap",
            {"document": document, "question": question, "answer": answer, **options},
        )


# --------------------------------------------------------------------------- #
# Heatmap -> dense matrix / per-statement mass                                #
# --------------------------------------------------------------------------- #


@dataclass
class Heatmap:
    """A parsed heatmap: dense matrix plus token offset maps into both strings."""

    matrix: np.ndarray  # [answer_tokens, document_tokens], unnormalized weights
    answer_offsets: list[tuple[int, int]]  # char [start,end) per answer token
    document_offsets: list[tuple[int, int]]  # char [start,end) per document token

    @classmethod
    def from_response(cls, sparse: dict) -> "Heatmap":
        matrix = np.zeros(sparse["shape"], dtype=float)
        matrix[sparse["row"], sparse["col"]] = sparse["data"]
        return cls(
            matrix=matrix,
            answer_offsets=[tuple(o) for o in sparse["answer_offsets"]],
            document_offsets=[tuple(o) for o in sparse["document_offsets"]],
        )

    # -- token selection ------------------------------------------------ #
    def answer_token_rows(self, start: int, end: int) -> list[int]:
        """Answer token indices overlapping the char span [start, end)."""
        return [
            i
            for i, (s, e) in enumerate(self.answer_offsets)
            if e > start and s < end
        ]

    def statement_mass(self, start: int, end: int) -> np.ndarray:
        """Summed, then L1-normalized, mass over document tokens for a statement.

        Sums the heatmap rows for every answer token overlapping the statement's
        char span and normalizes to a distribution over document tokens. A
        statement whose answer tokens carry no mass returns an all-zero vector.
        """
        rows = self.answer_token_rows(start, end)
        if not rows:
            return np.zeros(self.matrix.shape[1])
        mass = self.matrix[rows].sum(axis=0)
        total = mass.sum()
        return mass / total if total > 0 else mass

    def mass_to_spans(
        self,
        start: int,
        end: int,
        threshold: float,
        merge_gap_tokens: int = 2,
        max_spans: int = 4,
    ) -> list[tuple[int, int, float]]:
        """Turn a statement's mass distribution into document char spans.

        Keeps document tokens whose per-token mass fraction >= `threshold`,
        merges runs separated by <= `merge_gap_tokens`, and returns up to
        `max_spans` spans (by descending mass) as (char_start, char_end, mass).
        The mass is the fraction of the statement's total mass inside the span —
        this doubles as the citation's confidence in Exp 3.
        """
        mass = self.statement_mass(start, end)
        if mass.sum() == 0:
            return []
        kept = np.where(mass >= threshold)[0]
        if len(kept) == 0:
            return []

        # merge contiguous / near-contiguous token runs
        runs: list[list[int]] = []
        for tok in kept:
            if runs and tok - runs[-1][-1] <= merge_gap_tokens:
                runs[-1].append(int(tok))
            else:
                runs.append([int(tok)])

        spans: list[tuple[int, int, float]] = []
        for run in runs:
            c_start = self.document_offsets[run[0]][0]
            c_end = self.document_offsets[run[-1]][1]
            span_mass = float(mass[run].sum())
            spans.append((c_start, c_end, span_mass))
        spans.sort(key=lambda s: s[2], reverse=True)
        return spans[:max_spans]

    def concentration(self, start: int, end: int) -> float:
        """How peaked a statement's mass is on the document (Exp 3 signal).

        Returns 1 - normalized entropy of the mass distribution: ~1 when the
        statement's mass concentrates on a few document tokens (grounded),
        ~0 when it is spread thin across the document (a hallmark of an answer
        recalled from parametric memory rather than read off the context).
        """
        mass = self.statement_mass(start, end)
        p = mass[mass > 0]
        if p.size <= 1:
            return 1.0 if p.size == 1 else 0.0
        entropy = -(p * np.log(p)).sum()
        return float(1.0 - entropy / np.log(p.size))


def token_count_from_offsets(offsets: Iterable[tuple[int, int]]) -> int:
    return sum(1 for _ in offsets)

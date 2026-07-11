"""TokenPath API client + attribution-mass helpers.

Two endpoints, mirroring the cookbook notebooks:

  POST /v1/attributions          span -> single best source span (+confidence)
  POST /v1/attributions/heatmap  raw sparse answer_token x document_token matrix

The heatmap is what the benchmark leans on: from it we can compute, for any
answer character span (a "statement"), how much attribution mass lands on each
region of the document. That mass distribution is what we threshold into
sentence-level citations.

Every call records wall-clock latency so the latency/$ column in the results
table comes from real measurements, never estimates. Timing is captured with
time.perf_counter at the call site (see common/timing.py for aggregation);
this module returns the elapsed seconds alongside each response.
"""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass, field
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
    def _post(self, path: str, payload: dict, max_retries: int = 6) -> Timed:
        """POST with backoff on 429/5xx. `seconds` measures only the successful
        attempt's wall-clock (retry sleeps are excluded) so the latency column
        reflects the real heatmap call, not client-side rate-limit waiting."""
        last: Exception | None = None
        for attempt in range(max_retries):
            t0 = time.perf_counter()
            try:
                resp = self._session.post(
                    f"{self.api_url}{path}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                seconds = time.perf_counter() - t0
            except requests.RequestException as exc:
                last = exc
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last = TokenPathError(resp.status_code, resp.text,
                                      resp.headers.get("x-request-id"))
                if attempt == max_retries - 1:
                    break
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() \
                    else min(2 ** attempt, 30)
                time.sleep(delay)
                continue
            if not resp.ok:
                raise TokenPathError(
                    resp.status_code, resp.text, resp.headers.get("x-request-id")
                )
            return Timed(resp.json(), seconds)
        raise last  # exhausted retries

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
    # Cache: id(sentence_spans) -> token->sentence index array, so the constant
    # token->sentence mapping is built once per document, not per statement.
    _tok_sent_cache: dict = field(default_factory=dict, repr=False, compare=False)

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
        The mass is the fraction of the statement's total mass inside the span.
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

    # -- sentence-level citations (LongBench-Cite's native unit) --------- #
    def _token_sentence_index(
        self, doc_sentence_spans: Sequence[Sequence[int]]
    ) -> np.ndarray:
        """Map each document token to the index of the sentence it falls in.

        LongBench-Cite scores citations at the sentence level, so we assign every
        document token to a source sentence (by its char midpoint) and later sum
        attribution mass per sentence. Tokens before the first sentence get -1
        (dropped). The mapping is constant for a document, so it is memoized on
        the heatmap keyed by the sentence-span list's identity.
        """
        key = id(doc_sentence_spans)
        cached = self._tok_sent_cache.get(key)
        if cached is not None:
            return cached
        starts = [int(s) for s, _ in doc_sentence_spans]
        tok_sent = np.full(len(self.document_offsets), -1, dtype=int)
        for i, (cs, ce) in enumerate(self.document_offsets):
            mid = (cs + ce) / 2
            j = bisect.bisect_right(starts, mid) - 1  # last sentence starting <= mid
            if j >= 0:
                tok_sent[i] = j  # gaps (whitespace) attach to the preceding sentence
        self._tok_sent_cache[key] = tok_sent
        return tok_sent

    def mass_to_sentences(
        self,
        start: int,
        end: int,
        doc_sentence_spans: Sequence[Sequence[int]],
        threshold: float,
        max_spans: int = 4,
    ) -> list[tuple[int, int, float]]:
        """Aggregate a statement's token mass onto document SENTENCES.

        Sums the statement's normalized attribution mass within each document
        sentence and keeps the sentences whose share of the statement's total
        mass clears `threshold`, returning up to `max_spans` of them (by
        descending mass) as (char_start, char_end, mass) whole-sentence spans.

        This is the fair unit for LongBench-Cite: the benchmark's gold citations
        and every baseline (prompted, Citations API, published LongCite) cite
        whole sentences, so post-hoc attribution must too. The attribution itself
        is unchanged — we only report the cited sentence rather than the raw
        token run. Because a sentence pools many tokens' mass, the threshold is a
        well-scaled quantity here (unlike a per-token fraction on a long doc).
        """
        mass = self.statement_mass(start, end)
        if mass.sum() == 0 or len(doc_sentence_spans) == 0:
            return []
        tok_sent = self._token_sentence_index(doc_sentence_spans)
        sent_mass = np.zeros(len(doc_sentence_spans))
        valid = tok_sent >= 0
        np.add.at(sent_mass, tok_sent[valid], mass[valid])

        kept = np.where(sent_mass >= threshold)[0]
        if len(kept) == 0:
            return []
        order = kept[np.argsort(sent_mass[kept])[::-1]][:max_spans]
        out: list[tuple[int, int, float]] = []
        for j in order:
            ss, se = doc_sentence_spans[j]
            out.append((int(ss), int(se), float(sent_mass[j])))
        return out

def token_count_from_offsets(offsets: Iterable[tuple[int, int]]) -> int:
    return sum(1 for _ in offsets)

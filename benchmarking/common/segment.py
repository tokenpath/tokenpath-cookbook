"""Shared answer segmentation.

Every method attributes the SAME frozen answers, and every method must cite at
the SAME granularity, or the comparison is rigged. So segmentation lives here
and is called identically by all four methods and by the judge-input builder.

`statement_spans` returns sentence-level [start, end) char spans over the text,
matching the cookbook's `claim_spans`. A "statement" in LongBench-Cite terms is
one such span; each method's job is to attach supporting `cite` snippets to it.
"""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"[.!?][\"\')\]]*(?=\s|$)|\n")


def statement_spans(text: str) -> list[list[int]]:
    """Sentence-level [start, end) char spans, whitespace-trimmed."""
    raw: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        raw.append((start, m.end()))
        start = m.end()
    if start < len(text):
        raw.append((start, len(text)))

    spans: list[list[int]] = []
    for s, e in raw:
        segment = text[s:e]
        if segment.strip():
            left = len(segment) - len(segment.lstrip())
            right = len(segment) - len(segment.rstrip())
            spans.append([s + left, e - right])
    return spans


def statements(text: str) -> list[dict]:
    """Segmentation as {statement, span} records — the unit the judge scores."""
    return [{"statement": text[s:e], "span": [s, e]} for s, e in statement_spans(text)]

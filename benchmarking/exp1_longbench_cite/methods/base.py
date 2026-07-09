"""Common shape for the four citation methods.

A method takes one frozen example and returns a record in LongCite's judge
schema, plus the cost/latency we measured producing it:

    {
      "idx", "dataset", "query",
      "prediction": <the answer text the judge sees>,
      "statements": [{"statement": str, "span": [s,e],
                      "citation": [{"cite": str, "mass"?: float}]}],
      "method": str,
      "latency_s": float,       # wall-clock to attribute/cite this one answer
      "cost_usd": float,        # dollars to attribute/cite this one answer
      "extra": {...},           # method-specific (e.g. answer_preserved)
    }

`prediction` is the frozen answer verbatim for every method EXCEPT the Anthropic
Citations API, which regenerates — there `prediction` is the regenerated text and
`extra.answer_preserved` records how much of the frozen answer survived.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common import segment


@dataclass
class CitedAnswer:
    idx: str
    dataset: str
    query: str
    prediction: str
    statements: list[dict]
    method: str
    latency_s: float = 0.0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "idx": self.idx,
            "dataset": self.dataset,
            "query": self.query,
            "prediction": self.prediction,
            "statements": self.statements,
            "method": self.method,
            "latency_s": self.latency_s,
            "cost_usd": self.cost_usd,
            "extra": self.extra,
        }


def empty_statements(answer: str) -> list[dict]:
    """Segment an answer into statement records with no citations yet."""
    return [
        {"statement": s["statement"], "span": s["span"], "citation": []}
        for s in segment.statements(answer)
    ]


class Method:
    """Base class. Subclasses implement `cite(example, answer)`; the constructor
    receives whatever clients/config the method needs."""

    name = "base"

    def cite(self, example: dict, answer: str) -> CitedAnswer:  # pragma: no cover
        raise NotImplementedError

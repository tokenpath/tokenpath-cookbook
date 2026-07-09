"""Prompted-citation baseline: an LLM "add citations" pass over the frozen answer.

This is how most teams add citations without a dedicated system: hand the model
the document, the question, and the already-written answer, and ask it to attach,
for each sentence, the verbatim supporting snippet(s) from the document. One LLM
call per answer. The model does the linking; we do nothing but parse.

We ask for strict JSON (a list, one entry per numbered sentence, each with the
sentence's supporting quotes copied exactly from the document). Quotes that don't
occur verbatim in the document are dropped — the model doesn't get to invent
source text — which is the honest way to score a prompted citer.
"""

from __future__ import annotations

import json

from ... import config
from ...common.openrouter import OpenRouterClient
from ...common.segment import statements as segment_statements
from .base import CitedAnswer, Method

SYSTEM = (
    "You attach source citations to an answer that was written from a document. "
    "For each numbered sentence, return the exact substrings of the document that "
    "support it, copied VERBATIM (character-for-character) from the document. If a "
    "sentence is an introduction, transition, or inference that needs no citation, "
    "return an empty list for it. Never paraphrase a quote and never cite text that "
    "is not in the document. Respond with JSON only."
)


def build_prompt(document: str, query: str, sentences: list[str]) -> list[dict]:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    user = (
        f"<document>\n{document}\n</document>\n\n"
        f"<question>\n{query}\n</question>\n\n"
        f"<answer_sentences>\n{numbered}\n</answer_sentences>\n\n"
        'Return JSON: {"citations": [{"sentence": <int index>, '
        '"quotes": [<verbatim document substring>, ...]}, ...]} '
        "with one object per sentence index above."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


class PromptedMethod(Method):
    name = "prompted"

    def __init__(self, client: OpenRouterClient, model: str = config.PROMPTED_MODEL):
        self.client = client
        self.model = model

    def cite(self, example: dict, answer: str) -> CitedAnswer:
        document, query = example["context"], example["query"]
        segs = segment_statements(answer)
        sentences = [s["statement"] for s in segs]

        res = self.client.chat(
            self.model,
            build_prompt(document, query, sentences),
            temperature=0.0,
            max_tokens=2048,
        )
        by_sentence: dict[int, list[str]] = {}
        try:
            parsed = _parse_json(res.text)
            for entry in parsed.get("citations", []):
                idx = int(entry.get("sentence", -1))
                quotes = [q for q in entry.get("quotes", []) if isinstance(q, str)]
                by_sentence[idx] = quotes
        except Exception:
            by_sentence = {}  # unparseable -> no citations (scored honestly)

        statements = []
        for i, seg in enumerate(segs):
            quotes = by_sentence.get(i, [])
            # keep only quotes that occur verbatim in the document
            valid = [q for q in quotes if q and q in document]
            statements.append(
                {
                    "statement": seg["statement"],
                    "span": seg["span"],
                    "citation": [{"cite": q} for q in valid],
                }
            )

        return CitedAnswer(
            idx=example["idx"],
            dataset=example["dataset"],
            query=query,
            prediction=answer,
            statements=statements,
            method=self.name,
            latency_s=res.seconds,
            cost_usd=res.cost_usd,
            extra={
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "model": res.model,
            },
        )

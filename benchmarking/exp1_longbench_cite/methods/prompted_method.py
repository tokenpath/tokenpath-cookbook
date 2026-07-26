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

    def __init__(self, client: OpenRouterClient, model: str = config.PROMPTED_MODEL,
                 enable_thinking: bool | None = None, max_tokens: int = 2048,
                 name: str | None = None):
        self.client = client
        self.model = model
        # Exp 3 runs this method twice against a thinking-capable local model — once
        # plain, once with the thought channel on — and reports the better score, so
        # the prompted condition is not strawmanned by a knob we chose. Defaults
        # keep Exp 1's behavior (provider default, 2048 tokens).
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        if name:
            self.name = name

    @property
    def cache_signature(self) -> dict:
        return {
            "method": self.name,
            "model": self.model,
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
        }

    def cite(self, example: dict, answer: str,
             statement_spans_override: list[list[int]] | None = None) -> CitedAnswer:
        document, query = example["context"], example["query"]
        # Exp 4 Arm 2 scores a second-pass citer against inline citations and
        # attention on the SAME answer. That only isolates the citation source if
        # all three cite the same statements, so the caller can pin the boundaries.
        if statement_spans_override is not None:
            segs = [{"statement": answer[s:e], "span": [s, e]}
                    for s, e in statement_spans_override]
        else:
            segs = segment_statements(answer)
        sentences = [s["statement"] for s in segs]

        res = self.client.chat(
            self.model,
            build_prompt(document, query, sentences),
            temperature=0.0,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )
        by_sentence: dict[int, list[str]] = {}
        parse_ok = True
        try:
            parsed = _parse_json(res.text)
            for entry in parsed.get("citations", []):
                idx = int(entry.get("sentence", -1))
                quotes = [q for q in entry.get("quotes", []) if isinstance(q, str)]
                by_sentence[idx] = quotes
        except Exception:
            by_sentence = {}  # unparseable -> no citations (scored honestly)
            parse_ok = False

        # Count what we discard. "The model could not hold the JSON format" and
        # "the model quoted text that isn't in the document" are different failures
        # from "the model cited the wrong sentence", and a comparison that folds
        # them together silently overstates whatever it is compared against.
        n_quotes = n_dropped = 0
        statements = []
        for i, seg in enumerate(segs):
            quotes = by_sentence.get(i, [])
            # keep only quotes that occur verbatim in the document
            valid = [q for q in quotes if q and q in document]
            n_quotes += len(quotes)
            n_dropped += len(quotes) - len(valid)
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
                "format_ok": parse_ok,
                "n_quotes": n_quotes,
                "n_quotes_nonverbatim": n_dropped,
                "enable_thinking": self.enable_thinking,
                # Truncated mid-answer vs emitted-something-unparseable: with a
                # thought channel eating the window these are different failures,
                # and only one of them is fixed by a bigger budget.
                "truncated": res.finish_reason == "length",
                "finish_reason": res.finish_reason,
                # Stamped so judge_stage can tell a stale judgment from a current
                # one. Without it record_cache_signature() is None on both sides,
                # None == None compares "current", and a rerun after a prompt or
                # parser change silently re-reports the OLD scores.
                "cache_signature": self.cache_signature,
            },
        )

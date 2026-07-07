"""Anthropic Citations API baseline — reproduction mode.

Unlike the other three, this method does not attribute the frozen answer; it
REGENERATES an answer with citations enabled, because that is the only mode the
Citations API offers. We prompt it to reproduce the frozen answer as closely as
possible while citing, then:

  1. segment the regenerated answer and attach each output sentence's cited
     document text (the API returns exact `cited_text` char spans), and
  2. report an **answer-preservation rate**: how close the regenerated answer is
     to the frozen one (difflib ratio). The plan calls this asymmetry a finding:
     regeneration means you no longer ship the answer you validated. It is a
     first-class column, not a footnote.

This is the one baseline that cannot go through OpenRouter — Citations is a
proprietary Anthropic feature — so it talks to api.anthropic.com directly and is
skipped (with a logged note) when ANTHROPIC_API_KEY is unset.
"""

from __future__ import annotations

import difflib
import time

import requests

from ... import config
from ...common.segment import statement_spans
from ...common.segment import statements as segment_statements
from .base import CitedAnswer, Method

REPRO_PROMPT = (
    "Reproduce the following answer to the question as closely as possible — same "
    "facts, same wording where you can — but ground every factual claim in the "
    "document using citations. Do not add new claims.\n\n"
    "<question>\n{query}\n</question>\n\n"
    "<answer_to_reproduce>\n{answer}\n</answer_to_reproduce>"
)


class CitationsAPIMethod(Method):
    name = "citations_api"

    def __init__(self, api_key: str, model: str = config.CITATIONS_API_MODEL,
                 api_url: str | None = None, timeout: int = 180):
        self.api_key = api_key
        self.model = model
        self.api_url = (api_url or config.ANTHROPIC_API_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _call(self, document: str, query: str, answer: str) -> tuple[dict, float]:
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": document,
                            },
                            "title": "Source document",
                            "citations": {"enabled": True},
                        },
                        {
                            "type": "text",
                            "text": REPRO_PROMPT.format(query=query, answer=answer),
                        },
                    ],
                }
            ],
        }
        t0 = time.perf_counter()
        resp = self._session.post(
            f"{self.api_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        seconds = time.perf_counter() - t0
        resp.raise_for_status()
        return resp.json(), seconds

    def cite(self, example: dict, answer: str) -> CitedAnswer:
        document, query = example["context"], example["query"]
        data, seconds = self._call(document, query, answer)

        # Rebuild the regenerated answer text, tracking each text block's char
        # range in the output and the document text it cited.
        pieces: list[str] = []
        block_cites: list[tuple[int, int, list[str]]] = []  # (out_start, out_end, cited_texts)
        cursor = 0
        for block in data.get("content", []):
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            start = cursor
            pieces.append(text)
            cursor += len(text)
            cited = [
                c.get("cited_text", "")
                for c in (block.get("citations") or [])
                if c.get("cited_text")
            ]
            block_cites.append((start, cursor, cited))
        regenerated = "".join(pieces)

        # Attach citations to output sentences by char-span overlap.
        statements = []
        for span in statement_spans(regenerated):
            s, e = span
            cites: list[str] = []
            for bs, be, cited_texts in block_cites:
                if be > s and bs < e:
                    cites.extend(cited_texts)
            # de-dup, keep only citations verbatim in the document
            seen = set()
            valid = []
            for c in cites:
                if c and c in document and c not in seen:
                    seen.add(c)
                    valid.append({"cite": c})
            statements.append(
                {"statement": regenerated[s:e], "span": span, "citation": valid}
            )
        if not statements:  # empty regeneration -> one empty statement
            statements = [{"statement": regenerated, "span": [0, len(regenerated)],
                           "citation": []}]

        preserved = difflib.SequenceMatcher(None, answer, regenerated).ratio()
        usage = data.get("usage", {})
        return CitedAnswer(
            idx=example["idx"],
            dataset=example["dataset"],
            query=query,
            prediction=regenerated,  # judge scores the REGENERATED answer
            statements=statements,
            method=self.name,
            latency_s=seconds,
            cost_usd=0.0,  # Anthropic usage is in tokens; priced separately in the table note
            extra={
                "answer_preserved": round(preserved, 4),
                "regenerated": True,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "model": data.get("model", self.model),
            },
        )

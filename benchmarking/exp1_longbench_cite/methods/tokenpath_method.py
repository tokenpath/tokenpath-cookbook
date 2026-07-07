"""TokenPath method: attribution mass -> sentence citations.

One post-hoc call to /v1/attributions/heatmap over the frozen answer. For each
statement (sentence span), sum the heatmap rows for its answer tokens, normalize
to a mass distribution over document tokens, keep the document regions whose mass
fraction clears the tuned threshold, and use their source text as the citations.

This is the method the plan frames as "attribution mass -> sentence citations,
threshold tuned on val split only". Keeping multiple regions per statement is
what addresses the recall risk called out in the plan (multi-sentence support).

Cost: TokenPath bills per attributed token; we approximate a query's attributed
tokens as answer_tokens + document_tokens (all tokens the heatmap spans) and
price them at config.TOKENPATH_USD_PER_MTOK.
"""

from __future__ import annotations

from ... import config
from ...common.timing import tokenpath_cost_usd
from ...common.tokenpath import Heatmap, TokenPathClient
from .base import CitedAnswer, Method, empty_statements


class TokenPathMethod(Method):
    name = "tokenpath"

    def __init__(
        self,
        client: TokenPathClient,
        mass_threshold: float = config.TOKENPATH_MASS_THRESHOLD,
        usd_per_mtok: float = config.TOKENPATH_USD_PER_MTOK,
    ):
        self.client = client
        self.mass_threshold = mass_threshold
        self.usd_per_mtok = usd_per_mtok

    def cite(self, example: dict, answer: str) -> CitedAnswer:
        document, query = example["context"], example["query"]
        timed = self.client.heatmap(document, query, answer)
        hm = Heatmap.from_response(timed.value)

        statements = empty_statements(answer)
        for st in statements:
            s, e = st["span"]
            spans = hm.mass_to_spans(s, e, self.mass_threshold)
            st["citation"] = [
                {"cite": document[cs:ce], "mass": round(mass, 4),
                 "source_start": cs, "source_end": ce}
                for cs, ce, mass in spans
            ]
            st["concentration"] = round(hm.concentration(s, e), 4)

        attributed_tokens = len(hm.answer_offsets) + len(hm.document_offsets)
        return CitedAnswer(
            idx=example["idx"],
            dataset=example["dataset"],
            query=query,
            prediction=answer,
            statements=statements,
            method=self.name,
            latency_s=timed.seconds,
            cost_usd=tokenpath_cost_usd(attributed_tokens, self.usd_per_mtok),
            extra={
                "attributed_tokens": attributed_tokens,
                "answer_tokens": len(hm.answer_offsets),
                "document_tokens": len(hm.document_offsets),
                "mass_threshold": self.mass_threshold,
            },
        )

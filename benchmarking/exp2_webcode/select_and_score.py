"""Exp 2 core: attribution-guided citation selection over search results.

For one (query, provider, results) row:
  1. generate an answer from ALL the provider's results (one cheap generator),
  2. grade each result for groundedness against the known expected answer
     (does this result actually contain the information to answer correctly?),
  3. attribute the generated answer against the concatenated results and compute
     each result's attribution mass — how strongly the answer is associated with it,
  4. select results whose mass clears a single global threshold (identical across
     every provider — no per-provider tuning), and
  5. report citation precision for all returned results and for the selected set.

The experiment measures whether one shared attribution-mass selection rule yields
a more precise set of citations across search providers. It evaluates citation
selection over results that have already been returned, not retrieval itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..common.openrouter import OpenRouterClient
from ..common.tokenpath import Heatmap, TokenPathClient

RESULT_SEP = "\n\n=====\n\n"

GEN_SYSTEM = (
    "Answer the question using only the provided search results. Be concise and "
    "factual. Do not add citations or markers."
)

GROUNDED_PROMPT = """You are grading a search result. Given a question, the known correct answer, and a single search result, decide whether the result contains or supports the correct answer.
Respond with [[Yes]] if the result contains the information needed to give the correct answer, or [[No]] if it does not.
Answer in the format "Grounded: [[Yes/No]]".

<question>
{query}
</question>

<correct_answer>
{expected}
</correct_answer>

<result>
{result}
</result>"""


def concat_results(results: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Join result texts into one document; return per-result [start,end) ranges."""
    doc_parts, ranges, cursor = [], [], 0
    for i, r in enumerate(results):
        text = r["text"]
        if i > 0:
            cursor += len(RESULT_SEP)
            doc_parts.append(RESULT_SEP)
        ranges.append((cursor, cursor + len(text)))
        doc_parts.append(text)
        cursor += len(text)
    return "".join(doc_parts), ranges


@dataclass
class ProviderScore:
    qid: str
    provider: str
    n_results: int
    grounded: list[int]  # 1/0 per result
    result_mass: list[float]  # attribution mass per result (fraction of total)
    selected: list[int]  # result indices selected by the shared mass threshold
    latency_s: float
    tokenpath_seconds: float


class Exp2Scorer:
    def __init__(
        self,
        tp: TokenPathClient,
        llm: OpenRouterClient,
        generator_model: str = config.GENERATOR_MODEL,
        judge_model: str = config.JUDGE_MODEL,
        selection_threshold: float = config.WEBCODE_SELECTION_THRESHOLD,
    ):
        self.tp = tp
        self.llm = llm
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.selection_threshold = selection_threshold
        self.judge_cost_usd = 0.0
        self.gen_cost_usd = 0.0

    def _generate(self, query: str, doc: str) -> tuple[str, float]:
        res = self.llm.chat(
            self.generator_model,
            [
                {"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content":
                 f"<search_results>\n{doc}\n</search_results>\n\n"
                 f"<question>\n{query}\n</question>"},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        self.gen_cost_usd += res.cost_usd
        return res.text.strip(), res.seconds

    def _grounded(self, query: str, expected: str, result_text: str) -> int:
        import re

        res = self.llm.chat(
            self.judge_model,
            [{"role": "user", "content":
              GROUNDED_PROMPT.format(query=query, expected=expected, result=result_text)}],
            temperature=0.0,
            max_tokens=10,
        )
        self.judge_cost_usd += res.cost_usd
        m = re.findall(r"\[\[([a-zA-Z]+)\]\]", res.text)
        return 1 if (m and m[0].lower() == "yes") else 0

    def _result_mass(self, hm: Heatmap, answer_len: int, ranges: list[tuple[int, int]]) -> list[float]:
        """Fraction of the answer's total attribution mass landing in each result."""
        mass = hm.statement_mass(0, answer_len)  # normalized over document tokens
        out = []
        for (lo, hi) in ranges:
            s = 0.0
            for tok, (ts, te) in enumerate(hm.document_offsets):
                mid = (ts + te) / 2
                if lo <= mid < hi:
                    s += mass[tok]
            out.append(float(s))
        return out

    def score(self, row: dict) -> ProviderScore:
        results = row["results"]
        doc, ranges = concat_results(results)
        answer, gen_s = self._generate(row["query"], doc)

        grounded = [self._grounded(row["query"], row["expected_answer"], r["text"])
                    for r in results]

        timed = self.tp.heatmap(doc, row["query"], answer)
        hm = Heatmap.from_response(timed.value)
        result_mass = self._result_mass(hm, len(answer), ranges)
        selected = [
            i for i, mass in enumerate(result_mass)
            if mass >= self.selection_threshold
        ]

        return ProviderScore(
            qid=row["qid"], provider=row["provider"], n_results=len(results),
            grounded=grounded, result_mass=[round(m, 4) for m in result_mass],
            selected=selected, latency_s=gen_s, tokenpath_seconds=timed.seconds,
        )


def precision_before_after_selection(scores: list[ProviderScore]) -> dict:
    """Aggregate citation precision before and after selection, per provider.

    before = grounded fraction over ALL returned results (Exa's citation precision)
    after  = grounded fraction over the attribution-guided selected result set
    Rows where selection returns nothing are excluded from `after` and counted.
    """
    import numpy as np

    by_provider: dict[str, list[ProviderScore]] = {}
    for score in scores:
        by_provider.setdefault(score.provider, []).append(score)

    out = {}
    for provider, items in by_provider.items():
        before_vals, after_vals, empty = [], [], 0
        for score in items:
            if score.n_results:
                before_vals.append(sum(score.grounded) / score.n_results)
            if score.selected:
                after_vals.append(
                    sum(score.grounded[i] for i in score.selected) / len(score.selected)
                )
            else:
                empty += 1
        out[provider] = {
            "n_queries": len(items),
            "citation_precision_before_selection": (
                round(float(np.mean(before_vals)), 4) if before_vals else 0.0
            ),
            "citation_precision_after_selection": (
                round(float(np.mean(after_vals)), 4) if after_vals else 0.0
            ),
            "queries_with_no_selected_results": empty,
            "mean_results_returned": round(float(np.mean([s.n_results for s in items])), 2),
            "mean_results_selected": round(float(np.mean([len(s.selected) for s in items])), 2),
        }
    return out

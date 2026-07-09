"""Exp 2 core: TokenPath as a citation-precision filter over search results.

For one (query, provider, results) row:
  1. generate an answer from ALL the provider's results (one cheap generator),
  2. grade each result for groundedness against the known expected answer
     (does this result actually contain the information to answer correctly?),
  3. attribute the generated answer against the concatenated results and compute
     each result's attribution mass — how much the answer actually leaned on it,
  4. keep only results whose mass clears a single global threshold (identical
     across every provider — no per-provider tuning), and
  5. report citation precision BEFORE (all returned results) vs AFTER (only the
     results the answer actually used).

The finding we're testing: TokenPath doesn't retrieve — it drops the returned-
but-unused results that drag a provider's citation precision down, so precision
rises for every provider under the same filter.
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

CORRECT_PROMPT = """You are grading an answer for correctness. Given a question, the known correct answer, and a candidate answer, decide whether the candidate answer is correct (conveys the same key fact as the correct answer).
Respond with [[Yes]] if correct or [[No]] if not, in the format "Correct: [[Yes/No]]".

<question>
{query}
</question>

<correct_answer>
{expected}
</correct_answer>

<candidate_answer>
{candidate}
</candidate_answer>"""


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
    kept: list[int]  # result indices kept after the mass filter
    latency_s: float
    tokenpath_seconds: float
    # Exp 3 (memorization) signals — cheap add-ons computed on the same data:
    answer: str = ""
    answer_correct: int = 0  # judge: does the generated answer match expected?
    peak_mass: float = 0.0  # max single-result attribution mass (concentration)
    grounded_mass: float = 0.0  # attribution mass landing on grounded results


class Exp2Scorer:
    def __init__(
        self,
        tp: TokenPathClient,
        llm: OpenRouterClient,
        generator_model: str = config.GENERATOR_MODEL,
        judge_model: str = config.JUDGE_MODEL,
        mass_threshold: float = config.WEBCODE_MASS_THRESHOLD,
    ):
        self.tp = tp
        self.llm = llm
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.mass_threshold = mass_threshold
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

    def _correct(self, query: str, expected: str, candidate: str) -> int:
        import re

        res = self.llm.chat(
            self.judge_model,
            [{"role": "user", "content":
              CORRECT_PROMPT.format(query=query, expected=expected, candidate=candidate)}],
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
        kept = [i for i, m in enumerate(result_mass) if m >= self.mass_threshold]

        answer_correct = self._correct(row["query"], row["expected_answer"], answer)
        peak_mass = max(result_mass) if result_mass else 0.0
        grounded_mass = sum(m for m, g in zip(result_mass, grounded) if g)

        return ProviderScore(
            qid=row["qid"], provider=row["provider"], n_results=len(results),
            grounded=grounded, result_mass=[round(m, 4) for m in result_mass],
            kept=kept, latency_s=gen_s, tokenpath_seconds=timed.seconds,
            answer=answer, answer_correct=answer_correct,
            peak_mass=round(float(peak_mass), 4), grounded_mass=round(float(grounded_mass), 4),
        )


def precision_before_after(scores: list[ProviderScore]) -> dict:
    """Aggregate citation precision before vs after the mass filter, per provider.

    before = grounded fraction over ALL returned results (Exa's citation precision)
    after  = grounded fraction over only the results the answer actually used
    Rows where the filter kept nothing are excluded from `after` and counted.
    """
    import numpy as np

    by_provider: dict[str, list[ProviderScore]] = {}
    for s in scores:
        by_provider.setdefault(s.provider, []).append(s)

    out = {}
    for provider, items in by_provider.items():
        before_vals, after_vals, empty = [], [], 0
        for s in items:
            if s.n_results:
                before_vals.append(sum(s.grounded) / s.n_results)
            if s.kept:
                after_vals.append(sum(s.grounded[i] for i in s.kept) / len(s.kept))
            else:
                empty += 1
        out[provider] = {
            "n_queries": len(items),
            "citation_precision_before": round(float(np.mean(before_vals)), 4) if before_vals else 0.0,
            "citation_precision_after": round(float(np.mean(after_vals)), 4) if after_vals else 0.0,
            "queries_with_no_kept_results": empty,
            "mean_results_returned": round(float(np.mean([s.n_results for s in items])), 2),
            "mean_results_kept": round(float(np.mean([len(s.kept) for s in items])), 2),
        }
    return out

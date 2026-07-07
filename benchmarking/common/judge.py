"""Citation-quality judge — a faithful port of THUDM/LongCite's auto_scorer.py.

The recall / precision / F1 definitions and the three judge prompts are
reproduced VERBATIM from the published harness
(github.com/THUDM/LongCite, LongBench-Cite/auto_scorer.py) so our numbers sit on
the same ruler as the published LongCite-8B / SelfCite rows. The only change is
the transport: `query_llm(... model=GPT_MODEL ...)` is replaced by an OpenRouter
call to a single, pinned judge model (config.JUDGE_MODEL), used identically for
our methods and for the re-judged anchor generations.

Metric recap (from the original):
  recall    per statement: if it has citations, judge support of the joined
            snippets (Fully=1 / Partial=0.5 / No=0); if it has no citations,
            score = 1 - needs_citation (a statement that needn't be cited counts
            as supported). Recall = mean over statements.
  precision per citation: judge relevance (Relevant=1 / Unrelevant=0).
            Precision = mean over all citations; 0 if there are no citations.
  f1        harmonic mean of recall and precision.

Input record schema (same as auto_scorer.get_citation_score):
  {"query": str, "prediction": str, "statements": [{"statement": str,
    "citation": [{"cite": str}, ...]}, ...]}
"""

from __future__ import annotations

import re

from .openrouter import ChatResult, OpenRouterClient

# --- prompts, verbatim from LongCite/LongBench-Cite/auto_scorer.py ---------- #

need_citation_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation.
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".\n\n{}
"""

support_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this statement is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the statement is supported by or extracted from the snippet. This applies only to cases where the statement and parts of the snippet are almost identical.
- [[Partially supported]] - More than half of the content in the statement is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the statement has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The statement is largely unrelated to the snippet, or most key points in the statement do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...". \n\n{}"""

relevant_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almostly unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...". \n\n{}"""


def _cat_qa_and_statement(question: str, answer: str, statement: str) -> str:
    return (
        f"<question>\n{question.strip()}\n</question>\n\n"
        f"<response>\n{answer.strip()}\n</response>\n\n"
        f"<sentence>\n{statement.strip()}\n</sentence>"
    )


def _cat_question_statement_context(question: str, statement: str, context: str) -> str:
    return (
        f"<question>\n{question.strip()}\n</question>\n\n"
        f"<statement>\n{statement.strip()}\n</statement>\n\n"
        f"<snippet>\n{context.strip()}\n</snippet>\n\n"
    )


def _parse_bracket(s: str) -> str | None:
    m = re.findall(r"\[\[([ /a-zA-Z]+)\]\]", s)
    return m[0] if m else None


def _need_citation_score(s: str):
    l = _parse_bracket(s)
    if l is None:
        return None
    return 1 if "yes" in l.lower() else 0


def _support_score(s: str):
    l = _parse_bracket(s)
    if l is None:
        return None
    l = l.lower()
    if "fully" in l:
        return 1
    if "partially" in l:
        return 0.5
    return 0


def _relevant_score(s: str):
    l = _parse_bracket(s)
    if l is None:
        return None
    return 0 if "unrelevant" in l.lower() else 1


class CitationJudge:
    """Scores citation records exactly like LongCite's auto_scorer, via OpenRouter."""

    def __init__(self, client: OpenRouterClient, model: str):
        self.client = client
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def _ask(self, prompt: str):
        """One judged query, retrying up to 5x (t=0 then t=1) until it parses."""
        last_out = ""
        for t in range(5):
            res: ChatResult = self.client.chat(
                self.model,
                [{"role": "user", "content": prompt}],
                temperature=0 if t == 0 else 1,
                max_tokens=10,
                stop="Analysis:",
            )
            self.prompt_tokens += res.prompt_tokens
            self.completion_tokens += res.completion_tokens
            self.cost_usd += res.cost_usd
            last_out = res.text
            yield last_out

    def _judge(self, prompt: str, parser):
        for out in self._ask(prompt):
            score = parser(out)
            if score is not None:
                return score, out
        return None, out  # unparseable after retries

    def need_citation(self, question, answer, sentence):
        prompt = need_citation_prompt_template.format(
            _cat_qa_and_statement(question, answer, sentence)
        )
        return self._judge(prompt, _need_citation_score)

    def is_support(self, question, statement, context):
        if context == "":
            return 0, "No matched citation"
        prompt = support_prompt_template.format(
            _cat_question_statement_context(question, statement, context)
        )
        return self._judge(prompt, _support_score)

    def is_relevant(self, question, statement, citation):
        prompt = relevant_prompt_template.format(
            _cat_question_statement_context(question, statement, citation)
        )
        return self._judge(prompt, _relevant_score)

    # -- aggregate metrics (mirror score_recall / score_precision) ------- #
    def score_recall(self, question, answer, statements_with_citations):
        scores = []
        for js in statements_with_citations:
            statement, citations = js["statement"], js["citation"]
            matched = [c["cite"] for c in citations]
            if matched:
                context = "\n\n".join(matched).strip()
                score, out = self.is_support(question, statement, context)
                js.update({"support_output": out, "support_score": score})
                scores.append(score if score is not None else 0)
            else:
                score, out = self.need_citation(question, answer, statement)
                js.update(
                    {"support_output": out,
                     "support_score": (1 - score) if score is not None else None}
                )
                scores.append((1 - score) if score is not None else 0)
        return float(sum(scores) / len(scores)) if scores else 0.0

    def score_precision(self, question, answer, statements_with_citations):
        scores = []
        for js in statements_with_citations:
            statement, citations = js["statement"], js["citation"]
            for c in citations:
                score, out = self.is_relevant(question, statement, c["cite"])
                c.update({"relevant_output": out, "relevant_score": score})
                scores.append(score if score is not None else 0)
        return float(sum(scores) / len(scores)) if scores else 0.0

    def get_citation_score(self, js: dict, max_statement_num: int | None = 40) -> dict:
        question = js["query"]
        answer = js["prediction"]
        statements = js["statements"]
        answer = re.sub(r"<cite>.*?</cite>", "", answer, flags=re.DOTALL)
        answer = answer.replace("<statement>", "").replace("</statement>", "")
        if max_statement_num and len(statements) > max_statement_num:
            statements = statements[:max_statement_num]
        recall = self.score_recall(question, answer, statements)
        precision = self.score_precision(question, answer, statements)
        js["citation_recall"] = recall
        js["citation_precision"] = precision
        js["citation_f1"] = (
            0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
        )
        return js

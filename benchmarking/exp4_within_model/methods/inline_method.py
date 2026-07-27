"""Arm 2 — single-pass inline citations from the model under test.

The model answers the question and cites in ONE generation, in LongCite's own
`<statement>...<cite>[[i][j]]</cite></statement>` format over numbered document
sentences. This is how a team actually ships cited answers from an open model, and
it is the condition-A variant nobody can call artificial: there is no second
"add citations" call, and the citations come from the same forward pass that wrote
the answer.

Why numbered document SENTENCES rather than free-form quotes: TokenPath pools
attribution mass per document sentence (LongBench-Cite's native citation unit), so
citing the same units is what makes Arm 2's two conditions commensurable. If the
prompted side quoted arbitrary substrings and the attention side emitted
sentences, the judge would be scoring two different granularities.

The statement boundaries the MODEL chose are canonical for this arm: we keep them
as the statement list and record each one's exact span in the cleaned answer, so
the paired attention run attributes the identical statements. Anything the model
emits outside a <statement> tag is dropped from the answer, which is the honest
reading — untagged prose carries no citation and the model was told the format.
"""

from __future__ import annotations

import re

from ...common.openrouter import OpenRouterClient
from ...common.segment import statement_spans
from ...exp1_longbench_cite.methods.base import CitedAnswer, Method

SYSTEM = (
    "You answer questions about a document and cite the document sentences you "
    "used. The document is given as numbered sentences, each tagged <C{i}>. "
    "Write your answer as a sequence of statements. Wrap EVERY statement like "
    "this:\n"
    "<statement>The statement text.<cite>[[3][7]]</cite></statement>\n"
    "where the numbers are the <C{i}> sentences that support that statement. If a "
    "statement needs no citation (an introduction, transition, or your own "
    "inference), use an empty cite: <cite></cite>. Cite only sentence numbers that "
    "appear in the document. Output statements only — no preamble, no other text."
)

_INDEX_RE = re.compile(r"\[(\d+)\]")
_CITE_RE = re.compile(r"<cite>(.*?)</cite>", re.DOTALL)
_OPEN_RE = re.compile(r"<statement>", re.DOTALL)
_CLOSE_RE = re.compile(r"</statement>", re.DOTALL)


def _split_statements(text: str) -> list[tuple[str, str]]:
    """[(statement_text, cite_blob)] parsed LENIENTLY.

    LongCite's format closes each statement with </statement>, but the model
    frequently omits the closing tag and just opens the next <statement>. Requiring
    the close scored perfectly usable output as zero citations, so a statement runs
    until whichever comes first: its </statement>, the next <statement>, or the end
    of the text. Being strict here would have measured our regex, not the model.
    """
    out: list[tuple[str, str]] = []
    opens = [m.end() for m in _OPEN_RE.finditer(text)]
    for i, start in enumerate(opens):
        next_open = opens[i + 1] - len("<statement>") if i + 1 < len(opens) else len(text)
        close = _CLOSE_RE.search(text, start)
        end = min(next_open, close.start() if close else len(text))
        body = text[start:end]
        cites = _CITE_RE.findall(body)
        stmt = _CITE_RE.sub("", body).strip()
        out.append((stmt, " ".join(cites)))
    return out


def number_document(document: str) -> tuple[str, list[str]]:
    """Document rendered as <C{i}> sentences, plus the sentence texts by index.

    Uses the shared segmenter, so the citable units are exactly the units the
    attention side pools mass over.
    """
    spans = statement_spans(document)
    sentences = [document[s:e] for s, e in spans]
    numbered = "".join(f"<C{i}>{text}" for i, text in enumerate(sentences))
    return numbered, sentences


def build_prompt(document: str, query: str) -> list[dict]:
    numbered, _ = number_document(document)
    user = (
        f"<document>\n{numbered}\n</document>\n\n"
        f"<question>\n{query}\n</question>\n\n"
        "Answer the question using the document, in the <statement>/<cite> format."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_inline(text: str, sentences: list[str],
                 finish_reason: str = "") -> tuple[str, list[dict], dict]:
    """(clean_answer, statement records, parse diagnostics).

    Diagnostics are reported as a first-class column: a small model's failure to
    hold the format is a real property of the prompted condition, but it must not
    be silently scored as "cited badly". `format_ok` says whether we recovered any
    tagged statement at all; `dropped_indices` counts citations pointing at
    sentences that do not exist (hallucinated indices), which we discard.

    `truncated` matters specially with a thought channel: if the model is cut off
    before it closes </think>, the leftover reasoning cannot be stripped and would
    be parsed as if it were the answer. That is a budget failure, not a citation
    result, and it is flagged rather than scored.
    """
    matches = _split_statements(text)
    truncated = finish_reason == "length"
    unclosed_thinking = "<think>" in text and "</think>" not in text
    diag = {
        "format_ok": bool(matches) and not truncated and not unclosed_thinking,
        "n_statements": len(matches),
        "dropped_indices": 0,
        "untagged_chars": 0,
        "truncated": truncated,
        "unclosed_thinking": unclosed_thinking,
    }
    if not matches:
        # Nothing parseable: no citations. Keep the raw text as the answer so the
        # judge still sees what the model said.
        cleaned = re.sub(r"</?statement>|<cite>.*?</cite>", "", text, flags=re.DOTALL)
        cleaned = cleaned.strip()
        spans = statement_spans(cleaned)
        return (
            cleaned,
            [
                {"statement": cleaned[s:e], "span": [s, e], "citation": []}
                for s, e in spans
            ],
            diag,
        )

    tagged_chars = sum(len(s) for s, _ in matches)
    diag["untagged_chars"] = max(0, len(text.strip()) - tagged_chars)

    parts: list[str] = []
    cite_sets: list[list[str]] = []
    for stmt, cites in matches:
        if not stmt:
            continue
        parts.append(stmt)
        picked: list[str] = []
        for raw in _INDEX_RE.findall(cites or ""):
            i = int(raw)
            if 0 <= i < len(sentences):
                picked.append(sentences[i])
            else:
                diag["dropped_indices"] += 1
        cite_sets.append(picked)

    # Build the cleaned answer by joining the model's own statements, tracking each
    # one's exact span so the paired attention run scores the identical statements.
    clean_parts: list[str] = []
    statements: list[dict] = []
    cursor = 0
    for stmt, picked in zip(parts, cite_sets):
        if clean_parts:
            clean_parts.append(" ")
            cursor += 1
        start = cursor
        clean_parts.append(stmt)
        cursor += len(stmt)
        statements.append(
            {
                "statement": stmt,
                "span": [start, cursor],
                "citation": [{"cite": c} for c in picked],
            }
        )
    return "".join(clean_parts), statements, diag


class InlineCiteMethod(Method):
    """Generates the answer and its citations in one call from the model itself."""

    name = "inline"

    def __init__(self, client: OpenRouterClient, model: str, max_tokens: int = 2048,
                 enable_thinking: bool | None = None, name: str | None = None):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        if name:
            self.name = name

    # Bump when the parser changes: cached records hold PARSED output, so a parser
    # fix must invalidate them or a rerun silently reports the old parse.
    PARSER_VERSION = 2

    @property
    def cache_signature(self) -> dict:
        return {
            "method": self.name,
            "model": self.model,
            "parser_version": self.PARSER_VERSION,
            "enable_thinking": self.enable_thinking,
        }

    def answer_and_cite(self, example: dict) -> CitedAnswer:
        document, query = example["context"], example["query"]
        _, sentences = number_document(document)

        res = self.client.chat(
            self.model,
            build_prompt(document, query),
            temperature=0.0,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )
        clean, statements, diag = parse_inline(res.text, sentences, res.finish_reason)

        return CitedAnswer(
            idx=example["idx"],
            dataset=example["dataset"],
            query=query,
            prediction=clean,
            statements=statements,
            method=self.name,
            latency_s=res.seconds,
            cost_usd=res.cost_usd,
            extra={
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "model": res.model,
                "n_doc_sentences": len(sentences),
                # See PromptedMethod: an unstamped record makes stale judgments
                # look current, which is how a parser fix can land and change
                # nothing in the reported numbers.
                "cache_signature": self.cache_signature,
                **diag,
            },
        )

    def cite(self, example: dict, answer: str) -> CitedAnswer:
        """This arm generates its own answer; `answer` is ignored by design."""
        return self.answer_and_cite(example)

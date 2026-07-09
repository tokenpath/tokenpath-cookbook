"""Experiment 1-native — fair attribution-QUALITY comparison on a naturally
generated cited answer.

Exp 1 freezes a no-citation answer and asks every method to attribute it; the
Citations API can't do that (it only cites text it writes), so there it runs in
an artificial "reproduce" mode. This experiment removes that artifact:

  1. Let the Anthropic Citations API answer the question NORMALLY
     (document + question -> a cited answer). Score its NATIVE citations.
  2. Take that answer's TEXT (citations are structured metadata, so the text is
     just the answer) and have TokenPath and the embedding baseline attribute
     that SAME text post-hoc. Score those.

Every method is then judged on the IDENTICAL set of statements (the Citations
API's own answer), isolating attribution quality from answer-preservation and
from any generator paraphrasing. All stages cache under results/exp1nat_* so the
expensive Sonnet-5 generation and TokenPath heatmaps are paid once.

TokenPath's mass threshold is inherited from Exp 1's val tuning (results/
exp1_threshold.json) — a fixed, reasonable operating point; pass --tokenpath-threshold
to override, or --note it if you re-tune later.
"""

from __future__ import annotations

import argparse
import difflib
import os
import time

import numpy as np
import requests

from .. import config
from ..common import env
from ..common.cite_len import backend as cite_len_backend
from ..common.cite_len import mean_citation_len
from ..common.io_utils import (load_done_ids, parallel_append, read_json,
                               read_jsonl, write_json)
from ..common.judge import CitationJudge
from ..common.segment import statement_spans
from ..common.segment import statements as segment_statements
from ..common.timing import MethodCost
from . import load_data
from .load_data import EXCLUDED_FROM_AVG
from .methods.base import CitedAnswer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FROZEN_PATH = os.path.join(RESULTS_DIR, "exp1nat_frozen.jsonl")


def _p(name: str, stage: str) -> str:
    return os.path.join(RESULTS_DIR, f"exp1nat_{stage}_{name}.jsonl")


# --------------------------------------------------------------------------- #
# Stage 1: Citations API answers normally; save answer text + native citations #
# --------------------------------------------------------------------------- #
NATURAL_PROMPT = (
    "Answer the following question using only the provided document. Write a "
    "clear, self-contained answer and cite the document for the factual claims "
    "you make.\n\n<question>\n{query}\n</question>"
)


def _call_citations_api(document: str, query: str, model: str, api_key: str,
                        api_url: str, timeout: int = 180) -> tuple[dict, float]:
    payload = {
        "model": model,
        "max_tokens": 2048,  # roomy enough for one-page summaries (avoid truncation)
        "messages": [{
            "role": "user",
            "content": [
                {"type": "document",
                 "source": {"type": "text", "media_type": "text/plain", "data": document},
                 "title": "Source document", "citations": {"enabled": True}},
                {"type": "text", "text": NATURAL_PROMPT.format(query=query)},
            ],
        }],
    }
    last: Exception | None = None
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{api_url}/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json=payload, timeout=timeout,
            )
            seconds = time.perf_counter() - t0
        except requests.RequestException as exc:
            last = exc
            time.sleep(min(2 ** attempt, 30))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last = RuntimeError(f"anthropic {resp.status_code}: {resp.text[:200]}")
            ra = resp.headers.get("retry-after")
            time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit()
                       else min(2 ** attempt, 30))
            continue
        resp.raise_for_status()
        return resp.json(), seconds
    raise last  # exhausted retries


def _parse_native(data: dict, document: str) -> tuple[str, list[dict]]:
    """Rebuild the answer text and attach each output sentence's cited doc text."""
    pieces: list[str] = []
    block_cites: list[tuple[int, int, list[str]]] = []
    cursor = 0
    for block in data.get("content", []):
        if block.get("type") != "text":
            continue
        text = block.get("text", "")
        start = cursor
        pieces.append(text)
        cursor += len(text)
        cited = [c.get("cited_text", "") for c in (block.get("citations") or [])
                 if c.get("cited_text")]
        block_cites.append((start, cursor, cited))
    answer = "".join(pieces)

    statements = []
    for span in statement_spans(answer):
        s, e = span
        cites: list[str] = []
        for bs, be, cited_texts in block_cites:
            if be > s and bs < e:
                cites.extend(cited_texts)
        seen, valid = set(), []
        for c in cites:
            if c and c in document and c not in seen:
                seen.add(c)
                valid.append({"cite": c})
        statements.append({"statement": answer[s:e], "span": span, "citation": valid})
    if not statements:
        statements = [{"statement": answer, "span": [0, len(answer)], "citation": []}]
    return answer, statements


def gen_native(examples: list[dict], model: str) -> None:
    """Generate the naturally-cited Sonnet answers; write frozen text + native cites."""
    key = env.anthropic_key()
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY required for the native Citations API run")
    api_url = config.ANTHROPIC_API_URL.rstrip("/")
    native_out = _p("citations_api", "cited")
    done = load_done_ids(FROZEN_PATH)
    todo = [e for e in examples if e["idx"] not in done]
    print(f"gen_native: {len(done)} cached, {len(todo)} to generate ({model})")

    in_price, out_price = config.ANTHROPIC_PRICE.get(model, (0.0, 0.0))

    def worker(e: dict):
        data, seconds = _call_citations_api(e["context"], e["query"], model, key, api_url)
        answer, statements = _parse_native(data, e["context"])
        usage = data.get("usage", {})
        it, ot = usage.get("input_tokens") or 0, usage.get("output_tokens") or 0
        cost = it / 1e6 * in_price + ot / 1e6 * out_price
        # frozen text (what TokenPath/embedding will attribute)
        from ..common.io_utils import append_jsonl
        append_jsonl(FROZEN_PATH, {
            "idx": e["idx"], "dataset": e["dataset"], "query": e["query"],
            "prediction": answer, "gen_model": data.get("model", model),
            "gen_input_tokens": it, "gen_output_tokens": ot, "gen_cost_usd": cost,
            "gen_seconds": seconds,
        })
        # native citations, as a ready-to-judge cited record
        return CitedAnswer(
            idx=e["idx"], dataset=e["dataset"], query=e["query"], prediction=answer,
            statements=statements, method="citations_api", latency_s=seconds,
            cost_usd=cost, extra={"input_tokens": it, "output_tokens": ot,
                                  "model": data.get("model", model)},
        ).to_record()

    # Anthropic tolerates moderate concurrency; the client retries are per-call here.
    parallel_append(native_out, todo, worker, workers=6, desc="gen_native")


# --------------------------------------------------------------------------- #
# Stage 2: TokenPath + embedding attribute the SAME answer text                #
# --------------------------------------------------------------------------- #
def cite_stage(name: str, method, examples: list[dict], frozen: dict[str, str]) -> str:
    out = _p(name, "cited")
    done = load_done_ids(out)
    todo = [e for e in examples if e["idx"] not in done and e["idx"] in frozen]
    print(f"[{name}] cite: {len(done)} cached, {len(todo)} to do")

    def worker(e: dict) -> dict:
        try:
            return method.cite(e, frozen[e["idx"]]).to_record()
        except Exception as exc:
            return {"idx": e["idx"], "dataset": e["dataset"], "query": e["query"],
                    "prediction": frozen[e["idx"]], "statements": [], "method": name,
                    "latency_s": 0.0, "cost_usd": 0.0, "extra": {"error": str(exc)[:300]}}

    workers = {"tokenpath": 4, "embedding": 4}.get(name, 8)  # embedding has a local reranker
    parallel_append(out, todo, worker, workers=workers, desc=f"cite:{name}")
    return out


# --------------------------------------------------------------------------- #
# Stage 3: judge everything with the same judge                               #
# --------------------------------------------------------------------------- #
def judge_stage(name: str) -> str:
    cited_path = _p(name, "cited")
    out = _p(name, "judged")
    done = load_done_ids(out)
    cited = [r for r in read_jsonl(cited_path) if r["idx"] not in done]
    print(f"[{name}] judge: {len(done)} cached, {len(cited)} to do")
    client = env.openrouter_client()

    def worker(rec: dict) -> dict:
        judge = CitationJudge(client, config.JUDGE_MODEL)
        scored = judge.get_citation_score(dict(rec), max_statement_num=40)
        scored["judge_cost_usd"] = judge.cost_usd
        return scored

    parallel_append(out, cited, worker, workers=12, desc=f"judge:{name}")
    return out


# --------------------------------------------------------------------------- #
# Stage 4: aggregate + table                                                  #
# --------------------------------------------------------------------------- #
def aggregate(name: str) -> dict:
    judged = read_jsonl(_p(name, "judged"))
    cited = {r["idx"]: r for r in read_jsonl(_p(name, "cited"))}
    by_ds: dict[str, list[dict]] = {}
    cost = MethodCost(name)
    for r in judged:
        by_ds.setdefault(r["dataset"], []).append(r)
        c = cited.get(r["idx"], {})
        cost.add(c.get("latency_s", 0.0), c.get("cost_usd", 0.0))

    def m(items, key):
        return float(np.mean([x[key] for x in items])) if items else 0.0

    per_dataset = {ds: {"n": len(items),
                        "citation_recall": round(m(items, "citation_recall"), 4),
                        "citation_precision": round(m(items, "citation_precision"), 4),
                        "citation_f1": round(m(items, "citation_f1"), 4)}
                   for ds, items in sorted(by_ds.items())}
    avg_ds = [ds for ds in per_dataset if ds not in EXCLUDED_FROM_AVG] or list(per_dataset)
    all_ds = list(per_dataset)

    def avg_over(dslist, key):
        vals = [per_dataset[ds][key] for ds in dslist]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    mean_len, n_cites = mean_citation_len(judged)
    summary = {
        "method": name, "n_examples": len(judged), "per_dataset": per_dataset,
        "avg_reported": {"datasets": avg_ds,
                         "citation_recall": avg_over(avg_ds, "citation_recall"),
                         "citation_precision": avg_over(avg_ds, "citation_precision"),
                         "citation_f1": avg_over(avg_ds, "citation_f1")},
        "avg_all_datasets": {"datasets": all_ds,
                             "citation_recall": avg_over(all_ds, "citation_recall"),
                             "citation_precision": avg_over(all_ds, "citation_precision"),
                             "citation_f1": avg_over(all_ds, "citation_f1")},
        "citation_length": {"mean": round(mean_len, 2), "n": n_cites, "unit": cite_len_backend()},
        "cost_latency": cost.summary(),
        "judge_cost_usd_total": round(sum(r.get("judge_cost_usd", 0.0) for r in judged), 4),
    }
    write_json(os.path.join(RESULTS_DIR, f"exp1nat_scores_{name}.json"), summary)
    return summary


LABEL = {"citations_api": "Citations API — NATIVE citations (Sonnet-5, as generated)",
         "tokenpath": "TokenPath — post-hoc attribution of the SAME answer",
         "embedding": "Embedding retrieve+rerank — of the SAME answer",
         "prompted": "Prompted add-citations — of the SAME answer"}


def make_table(methods: list[str]) -> None:
    rows = []
    for name in methods:
        p = os.path.join(RESULTS_DIR, f"exp1nat_scores_{name}.json")
        if not os.path.exists(p):
            continue
        s = read_json(p)
        r = s["avg_reported"]; cl = s["cost_latency"]
        rows.append(f"| {LABEL.get(name, name)} | {r['citation_recall']:.3f} | "
                    f"{r['citation_precision']:.3f} | {r['citation_f1']:.3f} | "
                    f"{s['citation_length']['mean']} {s['citation_length']['unit']} | "
                    f"{cl['latency_p50_s']} | ${cl['usd_per_query_mean']:.5f} |")
    table = ("## Experiment 1-native — attribution quality on the SAME naturally-cited answer\n\n"
             "All methods are judged on the identical statements (the Citations API's own answer).\n\n"
             "| Method | Citation R | Citation P | Citation F1 | Cite len | Latency p50 (s) | $/query |\n"
             "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    out = os.path.join(RESULTS_DIR, "exp1nat_table.md")
    with open(out, "w") as f:
        f.write(table)
    print(table)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description="Exp 1-native: quality on a naturally-cited answer")
    ap.add_argument("--methods", nargs="+",
                    default=["citations_api", "tokenpath", "embedding"])
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--citations-model", default=config.CITATIONS_API_MODEL)
    ap.add_argument("--tokenpath-threshold", type=float, default=None)
    args = ap.parse_args()

    examples = load_data.load_split(args.split, seed=args.seed, english_only=True,
                                    limit_per_dataset=args.limit_per_dataset)
    print(f"split={args.split} examples={len(examples)}")

    # Stage 1: native generation (also produces the frozen answer text + native cites)
    gen_native(examples, model=args.citations_model)
    frozen = {r["idx"]: r["prediction"] for r in read_jsonl(FROZEN_PATH)}

    print(f"TokenPath aggregation: {config.TOKENPATH_AGG}")

    for name in args.methods:
        if name == "citations_api":
            pass  # native cites already written by gen_native
        elif name == "tokenpath":
            from .methods.tokenpath_method import TokenPathMethod
            cite_stage(name, TokenPathMethod(env.tokenpath_client()), examples, frozen)
        elif name == "embedding":
            from .methods.embedding_method import EmbeddingMethod
            cite_stage(name, EmbeddingMethod(), examples, frozen)  # hosted OpenAI embeddings
        elif name == "prompted":
            from .methods.prompted_method import PromptedMethod
            cite_stage(name, PromptedMethod(env.openrouter_client(), config.PROMPTED_MODEL),
                       examples, frozen)
        else:
            raise ValueError(name)
        judge_stage(name)
        s = aggregate(name)
        print(f"[{name}] F1={s['avg_reported']['citation_f1']} "
              f"p50={s['cost_latency']['latency_p50_s']}s "
              f"${s['cost_latency']['usd_per_query_mean']}/q")

    make_table(args.methods)


if __name__ == "__main__":
    main()

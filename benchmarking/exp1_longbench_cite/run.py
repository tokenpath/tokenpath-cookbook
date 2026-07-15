"""Run Experiment 1 end to end for one or more methods.

Pipeline (each stage cached, resumable):
  1. load the split (test by default) and the frozen answers
  2. for each method: attach citations to every frozen answer   -> exp1_cited_<m>.jsonl
  3. judge every cited record with the pinned judge             -> exp1_judged_<m>.jsonl
  4. aggregate F1/R/P per dataset + latency/cost/method-extras   -> exp1_scores_<m>.json

Averaging follows eval_cite.py: multifieldqa_en / multifieldqa_zh are excluded
from the headline average (we also report the per-dataset breakdown and an
all-datasets average so nothing is hidden).

Usage:
  python -m benchmarking.exp1_longbench_cite.run --methods tokenpath prompted \
      --split test --limit-per-dataset 25
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from .. import config
from ..common import env
from ..common.cite_len import backend as cite_len_backend
from ..common.cite_len import mean_citation_len
from ..common.io_utils import (is_error_record, parallel_append, read_json,
                               read_jsonl, read_jsonl_latest,
                               record_cache_signature, write_json)
from ..common.judge import CitationJudge
from ..common.timing import MethodCost
from . import freeze_answers, load_data
from .load_data import EXCLUDED_FROM_AVG
from .methods.base import CitedAnswer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# --------------------------------------------------------------------------- #
# Method construction                                                         #
# --------------------------------------------------------------------------- #
def build_method(name: str, cfg: config.RunConfig):
    if name == "tokenpath":
        from .methods.tokenpath_method import TokenPathMethod

        return TokenPathMethod(
            env.tokenpath_client(),
            mass_threshold=cfg.tokenpath_mass_threshold,
        )
    if name == "prompted":
        from .methods.prompted_method import PromptedMethod

        return PromptedMethod(env.openrouter_client(), cfg.prompted_model)
    if name == "embedding":
        from .methods.embedding_method import EmbeddingMethod

        return EmbeddingMethod(cfg.embed_model)
    if name == "citations_api":
        from .methods.citations_api_method import CitationsAPIMethod

        key = env.anthropic_key()
        if not key:
            return None  # skipped, logged by caller
        return CitationsAPIMethod(key, cfg.citations_api_model)
    raise ValueError(f"unknown method {name}")


# --------------------------------------------------------------------------- #
# Stage 2: attach citations                                                   #
# --------------------------------------------------------------------------- #
def cite_stage(name: str, method, examples: list[dict], frozen: dict[str, str]) -> str:
    out = os.path.join(RESULTS_DIR, f"exp1_cited_{name}.jsonl")
    cached = {r["idx"]: r for r in read_jsonl_latest(out)}

    def signature_for(e: dict):
        fn = getattr(method, "cache_signature_for", None)
        return (
            fn(e, frozen[e["idx"]])
            if fn is not None
            else getattr(method, "cache_signature", None)
        )

    done = {
        e["idx"]
        for e in examples
        if e["idx"] in frozen
        and (rec := cached.get(e["idx"])) is not None
        and not is_error_record(rec)
        and record_cache_signature(rec) == signature_for(e)
    }
    todo = [e for e in examples if e["idx"] not in done and e["idx"] in frozen]
    print(f"[{name}] cite: {len(done)} cached, {len(todo)} to do")

    def worker(e: dict) -> dict:
        signature = signature_for(e)
        try:
            return method.cite(e, frozen[e["idx"]]).to_record()
        except Exception as exc:  # keep going; a failed example is logged, not fatal
            return {"idx": e["idx"], "dataset": e["dataset"], "query": e["query"],
                    "prediction": frozen[e["idx"]], "statements": [], "method": name,
                    "latency_s": 0.0, "cost_usd": 0.0,
                    "extra": {"error": str(exc)[:300], "cache_signature": signature}}

    # Embedding runs local torch inference (GIL-bound, not thread-safe) -> serial.
    # TokenPath's API is rate-limited (429s under high concurrency) -> modest pool
    # + client-side backoff. The OpenAI/Anthropic-backed methods tolerate more.
    workers = {"embedding": 1, "tokenpath": 4}.get(name, 8)
    parallel_append(out, todo, worker, workers=workers, desc=f"cite:{name}")

    requested = {e["idx"] for e in examples if e["idx"] in frozen}
    failures = [r for r in read_jsonl_latest(out)
                if r["idx"] in requested and is_error_record(r)]
    if failures:
        first = (failures[0].get("extra") or {}).get("error", "unknown error")
        raise RuntimeError(
            f"[{name}] cite failed for {len(failures)}/{len(requested)} example(s); "
            f"refusing to judge an incomplete run. Rerun retries only those failures. "
            f"First error: {first}"
        )
    return out


# --------------------------------------------------------------------------- #
# Stage 3: judge                                                              #
# --------------------------------------------------------------------------- #
def judge_stage(name: str, cited_path: str, cfg: config.RunConfig) -> str:
    out = os.path.join(RESULTS_DIR, f"exp1_judged_{name}.jsonl")
    judged_by_idx = {r["idx"]: r for r in read_jsonl_latest(out)}
    cited_records = read_jsonl_latest(cited_path)
    failures = [r for r in cited_records if is_error_record(r)]
    if failures:
        raise RuntimeError(
            f"[{name}] cited cache still contains {len(failures)} failed record(s); "
            "rerun the cite stage before judging"
        )

    def is_current(rec: dict) -> bool:
        cached = judged_by_idx.get(rec["idx"])
        return bool(
            cached
            and cached.get("judge_model") == cfg.judge_model
            and record_cache_signature(cached) == record_cache_signature(rec)
            and is_error_record(cached) == is_error_record(rec)
            and "citation_f1" in cached
        )

    cited = [r for r in cited_records if not is_current(r)]
    print(f"[{name}] judge: {len(cited_records) - len(cited)} cached, {len(cited)} to do")
    client = env.openrouter_client()  # shared client; each worker gets its own judge

    def worker(rec: dict) -> dict:
        # A fresh CitationJudge per record so its cost/token counters aren't shared
        # across threads; the underlying HTTP client (connection pool) is shared.
        judge = CitationJudge(client, cfg.judge_model)
        scored = judge.get_citation_score(dict(rec), max_statement_num=40)
        scored["judge_cost_usd"] = judge.cost_usd
        scored["judge_model"] = cfg.judge_model
        return scored

    parallel_append(out, cited, worker, workers=12, desc=f"judge:{name}")

    latest = {r["idx"]: r for r in read_jsonl_latest(out)}
    missing = [
        r["idx"] for r in cited_records
        if not (
            (cached := latest.get(r["idx"]))
            and cached.get("judge_model") == cfg.judge_model
            and record_cache_signature(cached) == record_cache_signature(r)
            and is_error_record(cached) == is_error_record(r)
            and "citation_f1" in cached
        )
    ]
    if missing:
        raise RuntimeError(
            f"[{name}] judge did not complete {len(missing)} record(s); rerun resumes "
            f"the missing IDs. First missing idx: {missing[0]}"
        )
    return out


# --------------------------------------------------------------------------- #
# Stage 4: aggregate                                                          #
# --------------------------------------------------------------------------- #
def aggregate(name: str, judged_path: str, cited_path: str, cfg: config.RunConfig) -> dict:
    judged = read_jsonl_latest(judged_path)
    cited = {r["idx"]: r for r in read_jsonl_latest(cited_path)}
    failures = [r for r in judged if is_error_record(r)] + [
        r for r in cited.values() if is_error_record(r)
    ]
    if failures:
        raise RuntimeError(
            f"[{name}] caches still contain {len(failures)} failed record(s); "
            "rerun cite and judge before aggregating"
        )
    stale = [
        r for r in judged
        if (source := cited.get(r["idx"])) is None
        or r.get("judge_model") != cfg.judge_model
        or record_cache_signature(r) != record_cache_signature(source)
    ]
    if stale or len(judged) != len(cited):
        raise RuntimeError(
            f"[{name}] cite/judge caches are incomplete or out of sync; "
            "rerun cite and judge before aggregating"
        )

    by_ds: dict[str, list[dict]] = {}
    cost = MethodCost(name)
    preserved: list[float] = []
    for r in judged:
        by_ds.setdefault(r["dataset"], []).append(r)
        c = cited.get(r["idx"], {})
        cost.add(c.get("latency_s", 0.0), c.get("cost_usd", 0.0))
        ap = c.get("extra", {}).get("answer_preserved")
        if ap is not None:
            preserved.append(ap)

    def m(items, key):
        return float(np.mean([x[key] for x in items])) if items else 0.0

    per_dataset = {
        ds: {
            "n": len(items),
            "citation_recall": round(m(items, "citation_recall"), 4),
            "citation_precision": round(m(items, "citation_precision"), 4),
            "citation_f1": round(m(items, "citation_f1"), 4),
        }
        for ds, items in sorted(by_ds.items())
    }
    avg_ds = [ds for ds in per_dataset if ds not in EXCLUDED_FROM_AVG] or list(per_dataset)
    all_ds = list(per_dataset)

    def avg_over(dslist, key):
        vals = [per_dataset[ds][key] for ds in dslist]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    mean_len, n_cites = mean_citation_len(judged)
    summary = {
        "method": name,
        "n_examples": len(judged),
        "per_dataset": per_dataset,
        "avg_reported": {  # LongCite convention: excludes multifieldqa_en/zh
            "datasets": avg_ds,
            "citation_recall": avg_over(avg_ds, "citation_recall"),
            "citation_precision": avg_over(avg_ds, "citation_precision"),
            "citation_f1": avg_over(avg_ds, "citation_f1"),
        },
        "avg_all_datasets": {
            "datasets": all_ds,
            "citation_recall": avg_over(all_ds, "citation_recall"),
            "citation_precision": avg_over(all_ds, "citation_precision"),
            "citation_f1": avg_over(all_ds, "citation_f1"),
        },
        "citation_length": {"mean": round(mean_len, 2), "n": n_cites,
                            "unit": cite_len_backend()},
        "cost_latency": cost.summary(),
        "judge_cost_usd_total": round(sum(r.get("judge_cost_usd", 0.0) for r in judged), 4),
        "provenance": cfg.as_provenance(),
    }
    if preserved:
        summary["answer_preserved_mean"] = round(float(np.mean(preserved)), 4)
    write_json(os.path.join(RESULTS_DIR, f"exp1_scores_{name}.json"), summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run Experiment 1 (LongBench-Cite)")
    ap.add_argument("--methods", nargs="+",
                    default=["tokenpath", "prompted", "embedding", "citations_api"])
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-languages", action="store_true")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--run-date", default=config.DEFAULT_RUN_DATE)
    ap.add_argument("--skip-freeze", action="store_true")
    args = ap.parse_args()

    # Threshold: prefer the value tuned on val (written by tune_threshold.py).
    tuned = read_json(os.path.join(RESULTS_DIR, "exp1_threshold.json"), {})
    mass_thr = tuned.get("best_threshold", config.TOKENPATH_MASS_THRESHOLD)
    cfg = config.RunConfig(run_date=args.run_date, tokenpath_mass_threshold=mass_thr, seed=args.seed)

    examples = load_data.load_split(
        args.split, seed=args.seed, english_only=not args.all_languages,
        limit_per_dataset=args.limit_per_dataset,
    )
    print(f"split={args.split} examples={len(examples)} mass_threshold={mass_thr}")

    if not args.skip_freeze:
        freeze_answers.freeze(examples, model=cfg.generator_model)
    frozen = {r["idx"]: r["prediction"] for r in read_jsonl(freeze_answers.FROZEN_PATH)}

    summaries = {}
    for name in args.methods:
        method = build_method(name, cfg)
        if method is None:
            print(f"[{name}] SKIPPED (missing key/dependency)")
            continue
        cited_path = cite_stage(name, method, examples, frozen)
        judged_path = judge_stage(name, cited_path, cfg)
        summaries[name] = aggregate(name, judged_path, cited_path, cfg)
        s = summaries[name]
        print(f"[{name}] F1(reported)={s['avg_reported']['citation_f1']} "
              f"p50={s['cost_latency']['latency_p50_s']}s "
              f"${s['cost_latency']['usd_per_query_mean']}/q")
    write_json(os.path.join(RESULTS_DIR, "exp1_summaries.json"), summaries)


if __name__ == "__main__":
    main()

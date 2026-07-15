"""Tune TokenPath's mass threshold on the VAL split only.

Sweeps candidate thresholds, and for each one attaches TokenPath citations to the
frozen val answers, judges them, and records the average F1. The threshold with
the best val F1 is written to results/exp1_threshold.json, which run.py reads for
the TEST split. This is the val/test discipline the post commits to: the number
that picks the threshold is never the number we report.

To keep judge cost bounded, tuning defaults to a small per-dataset sample of val;
raise --limit-per-dataset for a finer estimate.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from .. import config
from ..common import aggregate as agg
from ..common import env
from ..common.io_utils import read_jsonl, write_json
from ..common.judge import CitationJudge
from ..common.segment import statement_spans
from ..common.tokenpath import Heatmap
from . import freeze_answers, load_data
from .load_data import EXCLUDED_FROM_AVG
from .methods.base import empty_statements

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _score_threshold(threshold, heatmaps, examples, frozen, client, judge_model, doc_sents):
    """Judge val answers cited at one threshold; return (reported-avg F1, cost).

    Each val answer is judged independently, so we fan the judging out across
    threads (a fresh CitationJudge per record; the HTTP client is shared)."""
    recs = []
    candidate_cfg = {**config.TOKENPATH_AGG, "threshold": threshold}
    for e in examples:
        idx = e["idx"]
        if idx not in heatmaps or idx not in frozen:
            continue
        hm = heatmaps[idx]
        answer = frozen[idx]
        statements = empty_statements(answer)
        for st in statements:
            s, en = st["span"]
            spans = agg.aggregate(
                hm,
                s,
                en,
                doc_sents[idx],
                candidate_cfg,
                answer_text=answer,
            )
            st["citation"] = [{"cite": e["context"][cs:ce]} for cs, ce, _ in spans]
        recs.append((e["dataset"],
                     {"query": e["query"], "prediction": answer, "statements": statements}))

    def judge_one(item):
        dataset, rec = item
        judge = CitationJudge(client, judge_model)
        scored = judge.get_citation_score(rec, max_statement_num=40)
        return dataset, scored["citation_f1"], judge.cost_usd

    by_ds: dict[str, list[float]] = {}
    cost = 0.0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for dataset, f1, c in ex.map(judge_one, recs):
            by_ds.setdefault(dataset, []).append(f1)
            cost += c
    ds_avgs = [np.mean(v) for ds, v in by_ds.items() if ds not in EXCLUDED_FROM_AVG] \
        or [np.mean(v) for v in by_ds.values()]
    return (float(np.mean(ds_avgs)) if ds_avgs else 0.0), cost


def main():
    ap = argparse.ArgumentParser(description="Tune TokenPath mass threshold on val")
    # Sentence-pooled mass shares are larger than per-token fractions, so the
    # sensible operating range sits higher than the old per-token thresholds.
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-languages", action="store_true")
    ap.add_argument("--limit-per-dataset", type=int, default=10)
    args = ap.parse_args()

    examples = load_data.load_split(
        "val", seed=args.seed, english_only=not args.all_languages,
        limit_per_dataset=args.limit_per_dataset,
    )
    freeze_answers.freeze(examples, model=config.GENERATOR_MODEL)
    frozen = {r["idx"]: r["prediction"] for r in read_jsonl(freeze_answers.FROZEN_PATH)}

    # One heatmap call per example, reused across all candidate thresholds.
    tp = env.tokenpath_client()
    heatmaps: dict[str, Heatmap] = {}
    doc_sents: dict[str, list] = {}  # idx -> document sentence spans (reused across thresholds)
    for e in tqdm(examples, desc="heatmaps(val)"):
        if e["idx"] not in frozen:
            continue
        timed = tp.heatmap(e["context"], e["query"], frozen[e["idx"]])
        heatmaps[e["idx"]] = Heatmap.from_response(timed.value)
        doc_sents[e["idx"]] = statement_spans(e["context"])

    client = env.openrouter_client()
    results = {}
    total_cost = 0.0
    for thr in args.thresholds:
        f1, cost = _score_threshold(
            thr, heatmaps, examples, frozen, client, config.JUDGE_MODEL, doc_sents
        )
        total_cost += cost
        results[str(thr)] = round(f1, 4)
        print(f"threshold {thr:>5}: val F1 = {f1:.4f}")

    best = max(results, key=results.get)
    out = {
        "best_threshold": float(best),
        "val_f1_by_threshold": results,
        "split": "val",
        "seed": args.seed,
        "judge_model": config.JUDGE_MODEL,
        "judge_cost_usd": round(total_cost, 4),
        "n_val_examples": len(heatmaps),
        "best_agg_cfg": {**config.TOKENPATH_AGG, "threshold": float(best)},
    }
    write_json(os.path.join(RESULTS_DIR, "exp1_threshold.json"), out)
    print(f"best threshold = {best} (val F1 {results[best]}) -> exp1_threshold.json")


if __name__ == "__main__":
    main()

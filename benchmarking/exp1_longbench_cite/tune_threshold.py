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

import numpy as np
from tqdm import tqdm

from .. import config
from ..common import env
from ..common.io_utils import read_jsonl, write_json
from ..common.judge import CitationJudge
from ..common.tokenpath import Heatmap
from . import freeze_answers, load_data
from .load_data import EXCLUDED_FROM_AVG
from .methods.base import empty_statements

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _score_threshold(threshold, heatmaps, examples, frozen, judge):
    """Judge val answers cited at one threshold; return reported-avg F1."""
    by_ds = {}
    for e in examples:
        idx = e["idx"]
        if idx not in heatmaps or idx not in frozen:
            continue
        hm = heatmaps[idx]
        answer = frozen[idx]
        statements = empty_statements(answer)
        for st in statements:
            s, en = st["span"]
            spans = hm.mass_to_spans(s, en, threshold)
            st["citation"] = [{"cite": e["context"][cs:ce]} for cs, ce, _ in spans]
        rec = {"query": e["query"], "prediction": answer, "statements": statements}
        scored = judge.get_citation_score(rec, max_statement_num=40)
        by_ds.setdefault(e["dataset"], []).append(scored["citation_f1"])
    ds_avgs = [np.mean(v) for ds, v in by_ds.items() if ds not in EXCLUDED_FROM_AVG] \
        or [np.mean(v) for v in by_ds.values()]
    return float(np.mean(ds_avgs)) if ds_avgs else 0.0


def main():
    ap = argparse.ArgumentParser(description="Tune TokenPath mass threshold on val")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.30])
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
    for e in tqdm(examples, desc="heatmaps(val)"):
        if e["idx"] not in frozen:
            continue
        timed = tp.heatmap(e["context"], e["query"], frozen[e["idx"]])
        heatmaps[e["idx"]] = Heatmap.from_response(timed.value)

    judge = CitationJudge(env.openrouter_client(), config.JUDGE_MODEL)
    results = {}
    for thr in args.thresholds:
        f1 = _score_threshold(thr, heatmaps, examples, frozen, judge)
        results[str(thr)] = round(f1, 4)
        print(f"threshold {thr:>5}: val F1 = {f1:.4f}")

    best = max(results, key=results.get)
    out = {
        "best_threshold": float(best),
        "val_f1_by_threshold": results,
        "split": "val",
        "seed": args.seed,
        "judge_model": config.JUDGE_MODEL,
        "judge_cost_usd": round(judge.cost_usd, 4),
        "n_val_examples": len(heatmaps),
    }
    write_json(os.path.join(RESULTS_DIR, "exp1_threshold.json"), out)
    print(f"best threshold = {best} (val F1 {results[best]}) -> exp1_threshold.json")


if __name__ == "__main__":
    main()

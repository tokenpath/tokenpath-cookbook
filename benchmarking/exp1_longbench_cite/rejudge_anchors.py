"""Re-judge published anchor generations with OUR judge.

The published LongCite-8B / SelfCite rows were scored with gpt-4o-2024-05-13. To
put every row on the same ruler, we re-score the anchor models' *own generations*
with the exact judge we use for our methods (config.JUDGE_MODEL). This is the
"re-judge anchors" path: same judge, same metric, different generations.

Input: a prediction file in LongCite's schema — a JSON list of
  {"idx", "dataset", "query", "prediction", "statements": [...]}
as produced by THUDM/LongCite's pred_sft.py / facebookresearch/SelfCite's rerank.
Point --preds at that file (obtain the generations from those repos; we do not
redistribute them). Output mirrors run.py's aggregate so make_table.py can place
the anchor row next to ours.

If you instead only have the *published summary numbers* (not the generations),
skip this and pass them to make_table.py as a greyed-out "published (GPT-4o
judge)" row — clearly labeled as a different judge.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .. import config
from ..common import env
from ..common.io_utils import append_jsonl, load_done_ids, read_json, read_jsonl, write_json
from ..common.judge import CitationJudge
from .load_data import EXCLUDED_FROM_AVG

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser(description="Re-judge anchor generations with our judge")
    ap.add_argument("--preds", required=True, help="LongCite-schema prediction JSON")
    ap.add_argument("--name", required=True, help="anchor label, e.g. longcite-8b")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    args = ap.parse_args()

    preds = read_json(args.preds)
    if args.limit_per_dataset:
        by_ds: dict[str, list] = {}
        for p in preds:
            bucket = by_ds.setdefault(p["dataset"], [])
            if len(bucket) < args.limit_per_dataset:
                bucket.append(p)
        preds = [p for v in by_ds.values() for p in v]

    judged_path = os.path.join(RESULTS_DIR, f"exp1_judged_anchor_{args.name}.jsonl")
    done = load_done_ids(judged_path)
    judge = CitationJudge(env.openrouter_client(), config.JUDGE_MODEL)
    for p in preds:
        if p["idx"] in done:
            continue
        rec = {
            "idx": p["idx"],
            "dataset": p["dataset"],
            "query": p["query"],
            "prediction": p["prediction"],
            "statements": p["statements"],
        }
        scored = judge.get_citation_score(rec, max_statement_num=40)
        append_jsonl(judged_path, scored)

    judged = read_jsonl(judged_path)
    by_ds = {}
    for r in judged:
        by_ds.setdefault(r["dataset"], []).append(r)

    def m(items, key):
        return round(float(np.mean([x[key] for x in items])), 4) if items else 0.0

    per_dataset = {ds: {k: m(items, k) for k in
                        ("citation_recall", "citation_precision", "citation_f1")}
                   for ds, items in sorted(by_ds.items())}
    avg_ds = [ds for ds in per_dataset if ds not in EXCLUDED_FROM_AVG] or list(per_dataset)
    summary = {
        "method": f"anchor:{args.name}",
        "note": "anchor generations re-judged with our judge (same ruler as our rows)",
        "n_examples": len(judged),
        "per_dataset": per_dataset,
        "avg_reported": {k: round(float(np.mean([per_dataset[ds][k] for ds in avg_ds])), 4)
                         for k in ("citation_recall", "citation_precision", "citation_f1")},
        "judge_model": config.JUDGE_MODEL,
        "judge_cost_usd": round(judge.cost_usd, 4),
    }
    write_json(os.path.join(RESULTS_DIR, f"exp1_scores_anchor_{args.name}.json"), summary)
    print(f"[anchor:{args.name}] reported F1 = {summary['avg_reported']['citation_f1']}")


if __name__ == "__main__":
    main()

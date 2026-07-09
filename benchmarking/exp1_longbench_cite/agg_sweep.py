"""Offline aggregation sweep — hill-climb TokenPath's citation quality.

Uses the cached raw heatmaps (cache_heatmaps.py), so trying a new aggregation
strategy costs nothing to compute — we only pay the judge. Each heatmap is loaded
ONCE and every candidate config is applied to it in memory, then each config's
citations are judged (parallel, cached). Iterate on the error dev set for signal;
validate the winners on the full 480.

  # build the error dev set (examples baseline TokenPath does worst on):
  python -m benchmarking.exp1_longbench_cite.agg_sweep --make-devset
  # sweep a batch on the dev set:
  python -m benchmarking.exp1_longbench_cite.agg_sweep --set dev --batch 1
  # validate specific configs on the full set:
  python -m benchmarking.exp1_longbench_cite.agg_sweep --set full --only baseline content_sqrt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .. import config
from ..common import aggregate as agg
from ..common import env
from ..common.io_utils import load_done_ids, parallel_append, read_json, read_jsonl, write_json
from ..common.judge import CitationJudge
from ..common.segment import statement_spans
from ..common.segment import statements as segment_statements
from ..common.tokenpath import Heatmap
from . import cache_heatmaps, load_data, native_run
from .load_data import EXCLUDED_FROM_AVG

SWEEP_DIR = os.path.join(native_run.RESULTS_DIR, "sweep")
DEVSET_PATH = os.path.join(native_run.RESULTS_DIR, "sweep_devset.json")

# --------------------------------------------------------------------------- #
# Candidate configs (grouped in batches so I can refine around winners)       #
# --------------------------------------------------------------------------- #
CONFIGS = {
    # batch 1 — one lever at a time + obvious combos
    "baseline":        {},
    "row_norm":        {"row_norm": True},
    "content":         {"atw": "content"},
    "sent_sqrt":       {"sent_norm": "sqrt"},
    "sent_len":        {"sent_norm": "len"},
    "max_pool":        {"pool": "max"},
    "content_sqrt":    {"atw": "content", "sent_norm": "sqrt"},
    "content_rn":      {"atw": "content", "row_norm": True},
    "content_sqrt_rn": {"atw": "content", "sent_norm": "sqrt", "row_norm": True},
    "thr10":           {"threshold": 0.10},
    "thr20":           {"threshold": 0.20},
    "merge":           {"merge_adjacent": True},
    "top1fb":          {"top1_fallback": True},
    "minconc15":       {"min_conc": 0.15},
    "content_sqrt_merge": {"atw": "content", "sent_norm": "sqrt", "merge_adjacent": True},
    "content_sqrt_top1": {"atw": "content", "sent_norm": "sqrt", "top1_fallback": True},
    "max_content":     {"pool": "max", "atw": "content"},

    # batch 2 — stack batch-1 winners (threshold + row_norm + content + merge),
    # push the threshold, and try relative (adaptive) thresholds.
    "thr20_rn":        {"threshold": 0.20, "row_norm": True},
    "thr25":           {"threshold": 0.25},
    "thr30":           {"threshold": 0.30},
    "thr25_rn":        {"threshold": 0.25, "row_norm": True},
    "thr30_rn":        {"threshold": 0.30, "row_norm": True},
    "thr20_rn_content":{"threshold": 0.20, "row_norm": True, "atw": "content"},
    "thr20_rn_merge":  {"threshold": 0.20, "row_norm": True, "merge_adjacent": True},
    "thr20_rn_content_merge": {"threshold": 0.20, "row_norm": True, "atw": "content",
                              "merge_adjacent": True},
    "thr25_rn_content_merge": {"threshold": 0.25, "row_norm": True, "atw": "content",
                              "merge_adjacent": True},
    "rel40":           {"threshold": 0.05, "rel_threshold": 0.40},
    "rel50":           {"threshold": 0.05, "rel_threshold": 0.50},
    "rel50_rn":        {"threshold": 0.05, "rel_threshold": 0.50, "row_norm": True},
    "rel40_rn_content":{"threshold": 0.05, "rel_threshold": 0.40, "row_norm": True,
                        "atw": "content"},
    "thr20_rn_ms3":    {"threshold": 0.20, "row_norm": True, "max_spans": 3},
    "thr20_rn_ms2":    {"threshold": 0.20, "row_norm": True, "max_spans": 2},

    # batch 3 — push threshold past 0.30 and combine thr30_rn with mild helpers
    "thr35_rn":        {"threshold": 0.35, "row_norm": True},
    "thr40_rn":        {"threshold": 0.40, "row_norm": True},
    "thr30_rn_content":{"threshold": 0.30, "row_norm": True, "atw": "content"},
    "thr30_rn_merge":  {"threshold": 0.30, "row_norm": True, "merge_adjacent": True},
    "thr30_rn_content_merge": {"threshold": 0.30, "row_norm": True, "atw": "content",
                              "merge_adjacent": True},
    "thr35_rn_merge":  {"threshold": 0.35, "row_norm": True, "merge_adjacent": True},
    "thr30_rn_ms3":    {"threshold": 0.30, "row_norm": True, "max_spans": 3},
    "thr30_rn_ms6":    {"threshold": 0.30, "row_norm": True, "max_spans": 6},
}
BATCHES = {
    1: [k for k in CONFIGS if k in ("baseline", "row_norm", "content", "sent_sqrt",
        "sent_len", "max_pool", "content_sqrt", "content_rn", "content_sqrt_rn",
        "thr10", "thr20", "merge", "top1fb", "minconc15", "content_sqrt_merge",
        "content_sqrt_top1", "max_content")],
    2: ["baseline", "thr20", "row_norm", "content_rn",  # carry best batch-1 refs
        "thr20_rn", "thr25", "thr30", "thr25_rn", "thr30_rn", "thr20_rn_content",
        "thr20_rn_merge", "thr20_rn_content_merge", "thr25_rn_content_merge",
        "rel40", "rel50", "rel50_rn", "rel40_rn_content", "thr20_rn_ms3", "thr20_rn_ms2"],
    3: ["baseline", "thr30_rn", "thr35_rn", "thr40_rn", "thr30_rn_content",
        "thr30_rn_merge", "thr30_rn_content_merge", "thr35_rn_merge",
        "thr30_rn_ms3", "thr30_rn_ms6"],
    # full-set validation of the threshold ladder + best combos
    99: ["baseline", "thr20_rn", "thr25_rn", "thr30_rn", "thr35_rn",
         "thr30_rn_merge", "thr30_rn_content_merge"],
}


# --------------------------------------------------------------------------- #
def _idxs_for(which: str) -> list:
    frozen_idx = [r["idx"] for r in read_jsonl(native_run.FROZEN_PATH)]
    if which == "full":
        return frozen_idx
    dev = read_json(DEVSET_PATH, {})
    return dev.get("idxs", frozen_idx)


def make_devset(f1_cutoff: float = 0.7):
    """Error dev set = examples where BASELINE TokenPath's F1 is weakest."""
    judged = read_jsonl(native_run._p("tokenpath", "judged"))
    scored = [(r["idx"], r["dataset"], r.get("citation_f1", 0.0)) for r in judged]
    hard = [idx for idx, ds, f1 in scored if f1 < f1_cutoff]
    write_json(DEVSET_PATH, {"idxs": hard, "f1_cutoff": f1_cutoff,
                             "n": len(hard), "n_total": len(scored)})
    print(f"dev error set: {len(hard)}/{len(scored)} examples with baseline F1 < {f1_cutoff}")
    return hard


# --------------------------------------------------------------------------- #
def build_cited(configs: dict, idxs: list) -> dict:
    """Load each heatmap once; apply every config. -> {name: {idx: record}}."""
    ex = {e["idx"]: e for e in load_data.load_split("test", seed=0, english_only=True)}
    frozen = {r["idx"]: r["prediction"] for r in read_jsonl(native_run.FROZEN_PATH)}
    out = {name: {} for name in configs}
    missing = 0
    for n, idx in enumerate(idxs):
        raw = cache_heatmaps.load_heatmap(idx)
        if raw is None or idx not in frozen:
            missing += 1
            continue
        hm = Heatmap.from_response(raw)
        doc = ex[idx]["context"]
        answer = frozen[idx]
        doc_sents = statement_spans(doc)
        segs = segment_statements(answer)
        for name, cfg in configs.items():
            statements = []
            for seg in segs:
                s, e = seg["span"]
                spans = agg.aggregate(hm, s, e, doc_sents, cfg, answer_text=answer)
                statements.append({"statement": seg["statement"], "span": seg["span"],
                                   "citation": [{"cite": doc[cs:ce], "mass": round(m, 4)}
                                                for cs, ce, m in spans]})
            out[name][idx] = {"idx": idx, "dataset": ex[idx]["dataset"], "query": ex[idx]["query"],
                              "prediction": answer, "statements": statements, "method": name}
        if (n + 1) % 100 == 0:
            print(f"  aggregated {n + 1}/{len(idxs)}")
    if missing:
        print(f"  ({missing} idxs missing heatmap/frozen — skipped)")
    return out


def judge_config(name: str, records: dict, which: str) -> list:
    """Judge one config's records (cached at sweep/<which>_<name>_judged.jsonl)."""
    os.makedirs(SWEEP_DIR, exist_ok=True)
    out = os.path.join(SWEEP_DIR, f"{which}_{name}_judged.jsonl")
    done = load_done_ids(out)
    todo = [rec for idx, rec in records.items() if idx not in done]
    client = env.openrouter_client()

    def worker(rec):
        judge = CitationJudge(client, config.JUDGE_MODEL)
        scored = judge.get_citation_score(dict(rec), max_statement_num=40)
        scored["judge_cost_usd"] = judge.cost_usd
        return scored

    if todo:
        parallel_append(out, todo, worker, workers=12, desc=f"judge:{which}:{name}")
    return read_jsonl(out)


def score(judged: list) -> dict:
    """Both a simple example-mean (dev signal) and the LongCite reported-avg."""
    ex_mean = lambda k: float(np.mean([r[k] for r in judged])) if judged else 0.0
    by_ds = {}
    for r in judged:
        by_ds.setdefault(r["dataset"], []).append(r)
    ds_f1 = {ds: float(np.mean([r["citation_f1"] for r in rs])) for ds, rs in by_ds.items()}
    rep_ds = [ds for ds in ds_f1 if ds not in EXCLUDED_FROM_AVG] or list(ds_f1)
    return {
        "n": len(judged),
        "f1": round(ex_mean("citation_f1"), 4),
        "recall": round(ex_mean("citation_recall"), 4),
        "precision": round(ex_mean("citation_precision"), 4),
        "f1_reported": round(float(np.mean([ds_f1[d] for d in rep_ds])), 4) if rep_ds else 0.0,
        "cost": round(sum(r.get("judge_cost_usd", 0.0) for r in judged), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-devset", action="store_true")
    ap.add_argument("--f1-cutoff", type=float, default=0.7)
    ap.add_argument("--set", choices=["dev", "full"], default="dev")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()

    if args.make_devset:
        make_devset(args.f1_cutoff)
        return

    names = args.only or (BATCHES.get(args.batch) if args.batch else list(CONFIGS))
    configs = {n: CONFIGS[n] for n in names}
    idxs = _idxs_for(args.set)
    print(f"sweep set={args.set} n_examples={len(idxs)} configs={len(configs)}")

    cited = build_cited(configs, idxs)
    results = {}
    for name in configs:
        judged = judge_config(name, cited[name], args.set)
        results[name] = score(judged)

    print(f"\n{'config':22} {'F1':>7} {'F1_rep':>7} {'recall':>7} {'prec':>7}  n")
    for name, s in sorted(results.items(), key=lambda kv: -kv[1]["f1"]):
        print(f"{name:22} {s['f1']:7.4f} {s['f1_reported']:7.4f} "
              f"{s['recall']:7.4f} {s['precision']:7.4f}  {s['n']}")
    write_json(os.path.join(SWEEP_DIR, f"leaderboard_{args.set}.json"), results)


if __name__ == "__main__":
    main()

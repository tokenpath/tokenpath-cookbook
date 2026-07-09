"""Run Experiment 2 (WebCode citation-precision filter).

For every (query, provider) row: generate, grade, attribute, filter, and record
a per-row score (cached, resumable). Then aggregate citation precision before vs
after the TokenPath mass filter, per provider, and write the scores + a note that
the SAME global threshold was used for every provider (no per-provider tuning).

Usage:
  python -m benchmarking.exp2_webcode.run --data benchmarking/data/webcode/webcode.jsonl
  python -m benchmarking.exp2_webcode.run --data .../sample.jsonl   # offline smoke test
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

from tqdm import tqdm

from .. import config
from ..common import env
from ..common.io_utils import append_jsonl, read_jsonl, write_json
from . import load_data
from .filter_and_score import Exp2Scorer, ProviderScore, precision_before_after

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SCORES_JSONL = os.path.join(RESULTS_DIR, "exp2_row_scores.jsonl")


def _row_key(row: dict) -> str:
    return f"{row['qid']}::{row['provider']}"


def main():
    ap = argparse.ArgumentParser(description="Run Experiment 2 (WebCode)")
    ap.add_argument("--data", default=load_data.DEFAULT_PATH)
    ap.add_argument("--threshold", type=float, default=config.WEBCODE_MASS_THRESHOLD)
    ap.add_argument("--run-date", default=config.DEFAULT_RUN_DATE)
    args = ap.parse_args()

    rows = load_data.load(args.data)
    done = {r["_key"] for r in read_jsonl(SCORES_JSONL) if "_key" in r}
    scorer = Exp2Scorer(
        env.tokenpath_client(), env.openrouter_client(),
        generator_model=config.GENERATOR_MODEL, judge_model=config.JUDGE_MODEL,
        mass_threshold=args.threshold,
    )

    todo = [r for r in rows if _row_key(r) not in done]
    print(f"exp2: {len(done)} cached rows, {len(todo)} to score, threshold={args.threshold}")
    for row in tqdm(todo, desc="exp2"):
        try:
            score = scorer.score(row)
            rec = asdict(score)
            rec["_key"] = _row_key(row)
            append_jsonl(SCORES_JSONL, rec)
        except Exception as exc:
            append_jsonl(SCORES_JSONL, {"_key": _row_key(row), "qid": row["qid"],
                                        "provider": row["provider"], "error": str(exc)[:300]})

    # Aggregate from all cached rows (skip error rows).
    cached = [r for r in read_jsonl(SCORES_JSONL) if "error" not in r]
    scores = [ProviderScore(
        qid=r["qid"], provider=r["provider"], n_results=r["n_results"],
        grounded=r["grounded"], result_mass=r["result_mass"], kept=r["kept"],
        latency_s=r["latency_s"], tokenpath_seconds=r["tokenpath_seconds"],
        answer=r.get("answer", ""), answer_correct=r.get("answer_correct", 0),
        peak_mass=r.get("peak_mass", 0.0), grounded_mass=r.get("grounded_mass", 0.0),
    ) for r in cached]

    agg = precision_before_after(scores)
    summary = {
        "experiment": "webcode_citation_precision_filter",
        "run_date": args.run_date,
        "mass_threshold": args.threshold,
        "note": "identical mass threshold across all providers — no per-provider tuning",
        "generator_model": config.GENERATOR_MODEL,
        "groundedness_judge_model": config.JUDGE_MODEL,
        "per_provider": agg,
        "gen_cost_usd": round(scorer.gen_cost_usd, 4),
        "judge_cost_usd": round(scorer.judge_cost_usd, 4),
    }
    write_json(os.path.join(RESULTS_DIR, "exp2_scores.json"), summary)
    for p, v in agg.items():
        print(f"{p:12} precision {v['citation_precision_before']:.3f} -> "
              f"{v['citation_precision_after']:.3f}  "
              f"(kept {v['mean_results_kept']}/{v['mean_results_returned']})")


if __name__ == "__main__":
    main()

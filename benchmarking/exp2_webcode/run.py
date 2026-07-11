"""Run Experiment 2 (WebCode attribution-guided citation selection).

For every (query, provider) row: generate, grade, attribute, select, and record
a per-row score (cached, resumable). Then aggregate citation precision for all
returned results and the attribution-guided selected set, per provider, and note
that the SAME global threshold was used for every provider (no per-provider tuning).

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
from .select_and_score import (
    Exp2Scorer,
    ProviderScore,
    precision_before_after_selection,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SCORES_JSONL = os.path.join(RESULTS_DIR, "exp2_row_scores.jsonl")


def _row_key(row: dict) -> str:
    return f"{row['qid']}::{row['provider']}"


def main():
    ap = argparse.ArgumentParser(description="Run Experiment 2 (WebCode)")
    ap.add_argument("--data", default=load_data.DEFAULT_PATH)
    ap.add_argument(
        "--selection-threshold",
        type=float,
        default=config.WEBCODE_SELECTION_THRESHOLD,
    )
    ap.add_argument("--run-date", default=config.DEFAULT_RUN_DATE)
    args = ap.parse_args()

    rows = load_data.load(args.data)
    cached_rows = read_jsonl(SCORES_JSONL)
    done = {
        r["_key"] for r in cached_rows
        if "_key" in r and ("selected" in r or "error" in r)
    }
    scorer = Exp2Scorer(
        env.tokenpath_client(), env.openrouter_client(),
        generator_model=config.GENERATOR_MODEL, judge_model=config.JUDGE_MODEL,
        selection_threshold=args.selection_threshold,
    )

    todo = [r for r in rows if _row_key(r) not in done]
    print(
        f"exp2: {len(done)} cached rows, {len(todo)} to score, "
        f"selection_threshold={args.selection_threshold}"
    )
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
    cached = [
        r for r in read_jsonl(SCORES_JSONL)
        if "error" not in r and "selected" in r
    ]
    scores = [ProviderScore(
        qid=r["qid"], provider=r["provider"], n_results=r["n_results"],
        grounded=r["grounded"], result_mass=r["result_mass"], selected=r["selected"],
        latency_s=r["latency_s"], tokenpath_seconds=r["tokenpath_seconds"],
    ) for r in cached]

    agg = precision_before_after_selection(scores)
    summary = {
        "experiment": "webcode_attribution_guided_citation_selection",
        "run_date": args.run_date,
        "selection_threshold": args.selection_threshold,
        "note": "identical selection threshold across all providers — no per-provider tuning",
        "generator_model": config.GENERATOR_MODEL,
        "result_support_judge_model": config.JUDGE_MODEL,
        "per_provider": agg,
        "gen_cost_usd": round(scorer.gen_cost_usd, 4),
        "judge_cost_usd": round(scorer.judge_cost_usd, 4),
    }
    write_json(os.path.join(RESULTS_DIR, "exp2_scores.json"), summary)
    for p, v in agg.items():
        print(
            f"{p:12} precision {v['citation_precision_before_selection']:.3f} -> "
            f"{v['citation_precision_after_selection']:.3f}  "
            f"(selected {v['mean_results_selected']}/{v['mean_results_returned']})"
        )


if __name__ == "__main__":
    main()

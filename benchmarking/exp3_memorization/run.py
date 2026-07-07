"""Experiment 3 — memorization detection (rides entirely on Exp 2's data).

WebCode is anti-memorization: the correct answer is supposed to require
retrieval. But models sometimes emit a correct answer with NO supporting result
present — they recalled it from parametric memory. TokenPath's attribution mass
should tell these apart: a grounded-correct answer concentrates mass on a
supporting result; a memorized-correct answer has nothing in the context to
attend to, so its mass is diffuse / low.

We reuse the per-row scores from exp2_row_scores.jsonl (no extra API calls) and
split the CORRECT answers into:
  - grounded-correct : answer_correct == 1 AND at least one grounded result
  - memorized-correct: answer_correct == 1 AND no grounded result
then compare their attribution-mass signal (peak single-result mass). We report a
separation AUC and write a histogram. Per the plan: if the two distributions
separate cleanly, this is one figure + two paragraphs; if AUC is near 0.5, it is
muddy — the script says so and we cut it from the post without further comment.
"""

from __future__ import annotations

import argparse
import os

from ..common.io_utils import read_jsonl, write_json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ROW_SCORES = os.path.join(RESULTS_DIR, "exp2_row_scores.jsonl")


def _auc(pos: list[float], neg: list[float]) -> float:
    """AUC that `pos` (grounded) scores higher than `neg` (memorized).

    Mann–Whitney U / (n_pos * n_neg): fraction of (pos, neg) pairs where the
    grounded row's signal exceeds the memorized row's (ties count 0.5). 1.0 = a
    threshold separates them perfectly; 0.5 = no separation.
    """
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _histogram(grounded: list[float], memorized: list[float], out_png: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"(skipping histogram: {exc})")
        return None

    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(grounded, bins=bins, alpha=0.7, label="grounded-correct", color="#6b5a9e", density=True)
    ax.hist(memorized, bins=bins, alpha=0.7, label="memorized-correct", color="#d98a5b", density=True)
    ax.set_xlabel("peak attribution mass on a single result")
    ax.set_ylabel("density")
    ax.set_title("Attribution mass separates grounded from memorized correct answers", loc="left")
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Run Experiment 3 (memorization detection)")
    ap.add_argument("--row-scores", default=ROW_SCORES)
    ap.add_argument("--signal", choices=["peak_mass", "grounded_mass"], default="peak_mass")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.row_scores) if "error" not in r]
    if not rows:
        raise SystemExit("no exp2_row_scores.jsonl — run benchmarking.exp2_webcode.run first")

    correct = [r for r in rows if r.get("answer_correct") == 1]
    grounded = [r[args.signal] for r in correct if any(r.get("grounded", []))]
    memorized = [r[args.signal] for r in correct if not any(r.get("grounded", []))]

    import numpy as np

    auc = _auc(grounded, memorized)
    summary = {
        "experiment": "memorization_detection",
        "signal": args.signal,
        "n_correct": len(correct),
        "n_grounded_correct": len(grounded),
        "n_memorized_correct": len(memorized),
        "grounded_mean": round(float(np.mean(grounded)), 4) if grounded else None,
        "memorized_mean": round(float(np.mean(memorized)), 4) if memorized else None,
        "separation_auc": round(auc, 4) if auc == auc else None,  # NaN-safe
        "verdict": (
            "insufficient data" if not (grounded and memorized)
            else "clean separation — include figure" if (auc >= 0.75 or auc <= 0.25)
            else "muddy — cut from post"
        ),
    }
    write_json(os.path.join(RESULTS_DIR, "exp3_scores.json"), summary)
    png = _histogram(grounded, memorized, os.path.join(RESULTS_DIR, "exp3_mass_hist.png"))
    print(summary)
    if png:
        print(f"wrote {png}")


if __name__ == "__main__":
    main()

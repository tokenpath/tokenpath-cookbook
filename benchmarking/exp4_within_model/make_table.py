"""Render Exp 4's within-model comparison as markdown.

Three views, because the headline number depends on which one you read and the
difference is not cosmetic:

  all          every test example. What a user of the model would actually get.
  well-formed  only examples where condition A produced parseable output. Flatters
               A by dropping the ones it could not handle.
  paired       BOTH conditions restricted to that same subset. The fair cut: the
               well-formed view otherwise hands A the easy examples while holding
               attention to all of them.

Condition A's format failures are printed as their own column, split into "could
not hold the format" and "ran out of completion budget". They are different
failures and only one of them says anything about citation quality — an early run
of this experiment reported a truncated row as though prompting simply cited
badly, which was wrong by ~0.25 F1.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .. import config
from ..exp1_longbench_cite import load_data
from ..exp1_longbench_cite.load_data import EXCLUDED_FROM_AVG

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# (label, prefix, condition-A row, paired attention row)
ARMS = [
    ("Arm 1 — frozen answer", "exp4", ["prompted", "prompted_thinking"], "tokenpath"),
    ("Arm 2 — single-pass inline vs second pass", "exp4inline",
     ["inline", "prompted_2pass"], "tokenpath"),
]


def _latest(path: str) -> dict:
    out: dict = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["idx"]] = rec
    return out


def _macro_f1(records: list[dict]) -> float:
    """LongCite's convention: mean per dataset, averaged over the reported ones."""
    by_ds: dict[str, list[float]] = {}
    for r in records:
        by_ds.setdefault(r["dataset"], []).append(r["citation_f1"])
    keep = [np.mean(v) for ds, v in by_ds.items() if ds not in EXCLUDED_FROM_AVG]
    if not keep:
        keep = [np.mean(v) for v in by_ds.values()]
    return float(np.mean(keep)) if keep else float("nan")


def _p(prefix: str, stage: str, name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{prefix}_{stage}_{name}.jsonl")


def build(split: str, seed: int) -> str:
    only = {e["idx"] for e in load_data.load_split(split, seed=seed, english_only=True)}
    thr = json.load(open(os.path.join(RESULTS_DIR, "exp4_threshold.json")))

    lines = [
        "# Exp 4 — within-model: attention vs the model's own words",
        "",
        f"Model under test: `{config.EXP4_MODEL}` (one model writes the answer, cites it, "
        "and is the model whose attention is read).",
        f"Split: `{split}` (n={len(only)}) · judge: `{config.JUDGE_MODEL}` · "
        f"attention mass threshold: {thr['best_threshold']} "
        f"(coarse val sweep, n={thr['n_val_examples']}, interior optimum).",
        "",
        "F1 is LongCite's reported average (excludes multifieldqa).",
        "",
    ]

    for arm_label, prefix, a_rows, b_row in ARMS:
        b_j = _latest(_p(prefix, "judged", b_row))
        lines += [f"## {arm_label}", "",
                  "| condition | n | F1 (all) | F1 (well-formed) | F1 paired | attention, same examples | gap (paired) | format ok | truncated |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for a_row in a_rows:
            a_j, a_c = _latest(_p(prefix, "judged", a_row)), _latest(_p(prefix, "cited", a_row))
            if not a_j:
                continue
            test_ids = [i for i in a_j if i in only]
            wf = {i for i in test_ids
                  if (a_c.get(i, {}).get("extra") or {}).get("format_ok") is True}
            paired = sorted(wf & set(b_j))

            n = len(test_ids)
            diags = [(a_c.get(i, {}).get("extra") or {}) for i in test_ids]
            fmt_ok = sum(1 for d in diags if d.get("format_ok")) / n if n else 0.0
            trunc = sum(1 for d in diags if d.get("truncated")) / n if n else 0.0

            f1_all = _macro_f1([a_j[i] for i in test_ids])
            f1_wf = _macro_f1([a_j[i] for i in sorted(wf)]) if wf else float("nan")
            f1_pair = _macro_f1([a_j[i] for i in paired]) if paired else float("nan")
            b_pair = _macro_f1([b_j[i] for i in paired]) if paired else float("nan")
            gap = b_pair - f1_pair

            lines.append(
                f"| A: {a_row} | {n} | {f1_all:.3f} | {f1_wf:.3f} | {f1_pair:.3f} | "
                f"{b_pair:.3f} | **{gap:+.3f}** | {fmt_ok:.0%} | {trunc:.0%} |"
            )

        b_ids = [i for i in b_j if i in only]
        if b_ids:
            lines.append(
                f"| B: attention (all) | {len(b_ids)} | {_macro_f1([b_j[i] for i in b_ids]):.3f} "
                "| — | — | — | — | n/a | n/a |"
            )
        lines.append("")

    # Cost/latency: same model, same GPU, so the ratio is the work ratio.
    lines += ["## Cost and latency (same weights, same GPU)", "",
              "| row | p50 latency | $/query |", "|---|---|---|"]
    for _, prefix, a_rows, b_row in ARMS:
        for name in a_rows + [b_row]:
            f = os.path.join(RESULTS_DIR, f"{prefix}_scores_{name}.json")
            if not os.path.exists(f):
                continue
            cl = json.load(open(f)).get("cost_latency", {})
            lines.append(f"| {prefix}/{name} | {cl.get('latency_p50_s', 0):.2f}s "
                         f"| ${cl.get('usd_per_query_mean', 0):.5f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Render Exp 4's within-model table")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "exp4_table.md"))
    args = ap.parse_args()
    table = build(args.split, args.seed)
    with open(args.out, "w") as fh:
        fh.write(table)
    print(table)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

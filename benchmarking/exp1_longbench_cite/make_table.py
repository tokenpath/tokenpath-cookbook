"""Render the Experiment 1 results table (Markdown).

Columns, per the plan: citation R / P / F1, citation length, wall-clock p50,
$/query, and answer-preservation (only meaningful for the regenerating Citations
API baseline). Latency lives in this table, not a separate section.

Rows:
  - our four post-hoc methods (from exp1_scores_<method>.json)
  - any re-judged anchors (exp1_scores_anchor_<name>.json) — same judge as ours
  - published reference rows (greyed, GPT-4o judge, DIFFERENT generations) so a
    reader can anchor against LongCite-8B / SelfCite without us implying parity.

The published numbers below are transcribed from the SelfCite repo's reproduced
BoN run (facebookresearch/SelfCite), judge gpt-4o-2024-05-13, base model
LongCite-llama3.1-8b. They are context, NOT a like-for-like comparison — the note
column says so.
"""

from __future__ import annotations

import argparse
import glob
import os

from ..common.io_utils import read_json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Published reference rows: reproduced SelfCite BoN, GPT-4o judge, LongCite-8B base.
# Averages here follow the repo's reported 'avg' (over the 5 task groups).
PUBLISHED = [
    {
        "label": "LongCite-8B + SelfCite BoN (published)",
        "recall": 0.759, "precision": 0.855, "f1": 0.782,
        "note": "published, GPT-4o judge, different generations — context only",
    },
]

METHOD_LABELS = {
    "tokenpath": "TokenPath (post-hoc attribution)",
    "prompted": "Prompted citation (LLM add-citations pass)",
    "embedding": "Embedding retrieve+rerank",
    "citations_api": "Anthropic Citations API (reproduction mode)",
}


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def load_our_rows() -> list[dict]:
    rows = []
    for name, label in METHOD_LABELS.items():
        path = os.path.join(RESULTS_DIR, f"exp1_scores_{name}.json")
        s = read_json(path)
        if not s:
            continue
        rep = s["avg_reported"]
        cl = s.get("cost_latency", {})
        rows.append({
            "label": label,
            "recall": rep["citation_recall"],
            "precision": rep["citation_precision"],
            "f1": rep["citation_f1"],
            "cite_len": s.get("citation_length", {}).get("mean"),
            "cite_len_unit": s.get("citation_length", {}).get("unit", ""),
            "p50": cl.get("latency_p50_s"),
            "usd": cl.get("usd_per_query_mean"),
            "answer_preserved": s.get("answer_preserved_mean"),
            "n": s.get("n_examples"),
            "note": "",
        })
    return rows


def load_anchor_rows() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "exp1_scores_anchor_*.json"))):
        s = read_json(path)
        rep = s["avg_reported"]
        rows.append({
            "label": s["method"].replace("anchor:", "") + " (re-judged, our judge)",
            "recall": rep["citation_recall"],
            "precision": rep["citation_precision"],
            "f1": rep["citation_f1"],
            "cite_len": None, "cite_len_unit": "", "p50": None, "usd": None,
            "answer_preserved": None, "n": s.get("n_examples"),
            "note": "anchor generations, our judge — comparable ruler",
        })
    return rows


def render(rows: list[dict]) -> str:
    header = (
        "| Method | Citation R | Citation P | Citation F1 | Cite len | "
        "Latency p50 (s) | $/query | Answer preserved | Note |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        cl = "—" if r["cite_len"] is None else f"{r['cite_len']:.1f} {r['cite_len_unit']}".strip()
        usd = "—" if r["usd"] is None else f"${r['usd']:.5f}"
        lines.append(
            f"| {r['label']} | {_fmt(r['recall'])} | {_fmt(r['precision'])} | "
            f"{_fmt(r['f1'])} | {cl} | {_fmt(r['p50'], 2)} | "
            f"{usd} | {_fmt(r['answer_preserved'])} | {r['note']} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Render Exp 1 results table")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "exp1_table.md"))
    args = ap.parse_args()

    rows = load_our_rows() + load_anchor_rows()
    for p in PUBLISHED:
        rows.append({
            "label": p["label"], "recall": p["recall"], "precision": p["precision"],
            "f1": p["f1"], "cite_len": None, "cite_len_unit": "", "p50": None,
            "usd": None, "answer_preserved": None, "n": None, "note": p["note"],
        })

    table = render(rows)
    title = "## Experiment 1 — LongBench-Cite (post-hoc attribution vs. the alternatives)\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(title + "\n" + table + "\n")
    print(title)
    print(table)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

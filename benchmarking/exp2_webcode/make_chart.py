"""Render Experiment 2: attribution-guided citation selection, per provider.

The grouped bar chart and Markdown table compare citation precision across all
returned results with precision in the attribution-guided selected set. Both
outputs land in results/.
"""

from __future__ import annotations

import argparse
import os

from ..common.io_utils import read_json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def render_table(agg: dict) -> str:
    header = (
        "| Provider | Citation precision (all results) | Citation precision "
        "(attribution-guided selection) | Δ | Mean results selected / returned |\n"
        "|---|---|---|---|---|"
    )
    lines = [header]
    for provider, v in sorted(agg.items()):
        before = v["citation_precision_before_selection"]
        after = v["citation_precision_after_selection"]
        delta = after - before
        lines.append(
            f"| {provider} | {before:.3f} | {after:.3f} | {delta:+.3f} | "
            f"{v['mean_results_selected']}/{v['mean_results_returned']} |"
        )
    return "\n".join(lines)


def render_chart(agg: dict, out_png: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # matplotlib optional
        print(f"(skipping chart: {exc})")
        return None

    providers = sorted(agg)
    before = [agg[p]["citation_precision_before_selection"] for p in providers]
    after = [agg[p]["citation_precision_after_selection"] for p in providers]

    x = np.arange(len(providers))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, before, w, label="all returned results", color="#b9a9e0")
    ax.bar(
        x + w / 2,
        after,
        w,
        label="attribution-guided selection",
        color="#6b5a9e",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in providers])
    ax.set_ylabel("Citation precision")
    ax.set_ylim(0, 1)
    ax.set_title(
        "WebCode citation precision — all results vs attribution-guided selection",
        loc="left",
    )
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Render Exp 2 chart + table")
    ap.add_argument("--scores", default=os.path.join(RESULTS_DIR, "exp2_scores.json"))
    args = ap.parse_args()

    summary = read_json(args.scores)
    if not summary:
        raise SystemExit("no exp2_scores.json — run benchmarking.exp2_webcode.run first")
    agg = summary["per_provider"]

    table = render_table(agg)
    md_path = os.path.join(RESULTS_DIR, "exp2_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## Experiment 2 — WebCode attribution-guided citation selection\n\n")
        f.write(f"_Single selection threshold {summary['selection_threshold']} across all "
                f"providers; no per-provider tuning._\n\n")
        f.write(table + "\n")
    print(table)
    png = render_chart(agg, os.path.join(RESULTS_DIR, "exp2_precision.png"))
    print(f"\nwrote {md_path}" + (f" and {png}" if png else ""))


if __name__ == "__main__":
    main()

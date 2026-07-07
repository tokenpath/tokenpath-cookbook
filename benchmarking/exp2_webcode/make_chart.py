"""Render the Experiment 2 figure + table: citation precision before vs after the
TokenPath mass filter, per provider.

One grouped bar chart (before vs after) and a Markdown table. The chart is the
positioning move: TokenPath sits next to all five providers as a precision layer,
competitor to none. Both outputs land in results/.
"""

from __future__ import annotations

import argparse
import os

from ..common.io_utils import read_json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def render_table(agg: dict) -> str:
    header = (
        "| Provider | Citation precision (before) | Citation precision "
        "(after TokenPath filter) | Δ | Mean results kept / returned |\n"
        "|---|---|---|---|---|"
    )
    lines = [header]
    for provider, v in sorted(agg.items()):
        before, after = v["citation_precision_before"], v["citation_precision_after"]
        delta = after - before
        lines.append(
            f"| {provider} | {before:.3f} | {after:.3f} | {delta:+.3f} | "
            f"{v['mean_results_kept']}/{v['mean_results_returned']} |"
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
    before = [agg[p]["citation_precision_before"] for p in providers]
    after = [agg[p]["citation_precision_after"] for p in providers]

    x = np.arange(len(providers))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, before, w, label="before", color="#b9a9e0")
    ax.bar(x + w / 2, after, w, label="after TokenPath filter", color="#6b5a9e")
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in providers])
    ax.set_ylabel("Citation precision")
    ax.set_ylim(0, 1)
    ax.set_title("WebCode citation precision — before vs after TokenPath filter", loc="left")
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
        f.write("## Experiment 2 — WebCode citation-precision filter\n\n")
        f.write(f"_Single mass threshold {summary['mass_threshold']} across all "
                f"providers; no per-provider tuning._\n\n")
        f.write(table + "\n")
    print(table)
    png = render_chart(agg, os.path.join(RESULTS_DIR, "exp2_precision.png"))
    print(f"\nwrote {md_path}" + (f" and {png}" if png else ""))


if __name__ == "__main__":
    main()

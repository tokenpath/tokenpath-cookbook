# Blog post materials — "How good are post-hoc citations?"

Everything needed to publish the Experiment-1 benchmark post.

## Files
- **`post.md`** — the blog post (Markdown). Self-contained prose + tables.
- **`data/`** — raw numbers for every chart (CSV + JSON), so charts can be
  regenerated in the site's own theme. The data is theme-agnostic.

## `data/` contents
| File | What it is |
|---|---|
| `main_comparison.csv` | The 4-method headline table: recall, precision, F1 (reported-avg + example-mean), citation length, cites/statement, latency p50, $/query. |
| `per_dataset_f1.csv` | F1 per dataset (gov_report / hotpotqa / longbench-chat / multifieldqa_en) × method. |
| `tokenpath_tuning_trajectory.csv` | TokenPath F1 across the four aggregation-tuning steps (0.722 → 0.785; original Llama-3.1-8B-era run — the resulting config is what the current 0.815 run uses). |
| `exp1_methods.json` | Full per-method detail (both F1 conventions, per-dataset F1, R/P, cite length, cites/stmt, latency, cost, kind) — the source of truth. |
| `sweep_leaderboard_dev.json` / `sweep_leaderboard_full.json` | Every aggregation config tried (dev error set + full-test validation), incl. the rejected ones. |

## Suggested charts (see `post.md` appendix for details)
1. Headline F1 bar chart — `main_comparison.csv` (`f1_reported`).
2. Cost/quality frontier scatter — F1 vs `$/query` (log x) and F1 vs latency — `main_comparison.csv`. (The key figure: TokenPath at the knee.)
3. Per-dataset grouped bars — `per_dataset_f1.csv`.
4. Tuning trajectory step chart — `tokenpath_tuning_trajectory.csv`.
5. Citation-length bar chart — `main_comparison.csv` (`cite_len_glm4`).

**Theme note:** use the site's brand palette and light/dark handling; do not hard-code
the placeholder colors. Highlight the TokenPath series/bar as the protagonist.

## Regenerating the underlying numbers
All numbers come from the harness under `benchmarking/`. The native same-answer
comparison is `python -m benchmarking.exp1_longbench_cite.native_run`; the
TokenPath aggregation sweep is `benchmarking/exp1_longbench_cite/agg_sweep.py`.
Per-example judge outputs and scores are in `benchmarking/results/` (and the
canonical final table is `benchmarking/results/exp1nat_table_final.md`).

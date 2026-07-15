# How good are post-hoc citations? Benchmarking TokenPath against generation-time citation

*Attributing an answer to its sources after the fact — from an open model's attention, on any model's output — matches generation-time citation quality (F1 0.815, ahead of the Citations API's 0.812; the prompted frontier LLM leads at 0.851), at ~5–6× lower latency and ~7× lower cost.*

> **Updated 2026-07-15.** We upgraded the open reference model the hosted API
> reads attention from — Llama-3.1-8B to **Qwen3.5-9B** — and re-ran TokenPath:
> same harness, same shared answers, same judge. TokenPath moved from F1 0.785
> to 0.815; the other three methods are unchanged. The pre-upgrade
> (Llama-3.1-8B) numbers are preserved in `data/exp1_methods.json` under
> `tokenpath_tuned_llama31_archived`.

---

## The problem

Everyone wants grounded answers with citations. Today you get them one of three ways:

1. **Prompt an LLM to cite** — hand the model the document, the question, and the answer, and ask it to attach supporting quotes ("add-citations pass").
2. **Regenerate with a citation API** — e.g. Anthropic's Citations API, which rewrites the answer while emitting citations inline.
3. **Retrieve** — embed the answer and search the source for similar passages.

All three either need the generating model to cooperate, cost a second frontier LLM call, or reshape the answer you already validated. **TokenPath does something different: it reads an open-source model's attention map *after* generation to attribute an answer to its source — on output from *any* model, with no regeneration.**

That's powerful, but it makes TokenPath awkward to benchmark: the alternatives all do a slightly different thing. So instead of a rigged apples-to-apples table, we measured post-hoc attribution against the three ways people *actually* get citations today, on a benchmark we didn't write.

**TL;DR:** post-hoc attention attribution now matches generation-time methods on quality (F1 0.815 vs the Citations API's 0.812; only a prompted frontier LLM is ahead at 0.851) and is well ahead of naive retrieval (0.62), while being ~6× faster, ~7× cheaper, needing no document index, and working on any model's output.

---

## The setup: one answer, four ways to cite it

We use **[LongBench-Cite](https://github.com/THUDM/LongCite)** (THUDM), a citation-quality benchmark over long documents. We take the English subset — `gov_report`, `hotpotqa`, `longbench-chat`, `multifieldqa_en` — split deterministically into 120 validation / **480 test** examples.

The key design choice is **fairness through a shared answer**. In our headline experiment:

1. We let the **Anthropic Citations API answer each question normally** (document + question → a cited answer from Claude Sonnet 5). This is a real, naturally-generated cited answer — no artificial "reproduce this text" prompting.
2. We take **that answer's text** and have every other method attribute *the same text* post-hoc: TokenPath, a prompted gpt-5.5 add-citations pass, and embedding retrieve+rerank.

So all four methods are judged on the **identical set of statements** — the same answer, sentence for sentence. That isolates *citation quality* from two confounds that muddy most comparisons: how the answer was phrased, and whether the method silently rewrote it.

Each method makes **one call per `(document, question, answer)` triple** — the whole answer at once, not per sentence — so latency and cost are directly comparable.

### The judge

Citations are scored by a **faithful port of LongCite's `auto_scorer.py`** — the same recall/precision/F1 definitions and prompts as the published benchmark, so our numbers sit on the same ruler. One consistent judge (`gemini-2.5-flash`) scores every method. It's reference-free: for each statement it asks an LLM "is this statement supported by the cited snippet?" (recall) and "is this cited snippet relevant?" (precision); F1 is their harmonic mean.

We report **reported-average F1** (the LongCite convention: mean over datasets, excluding `multifieldqa_en`) as the headline, matching published numbers.

---

## Results

**Experiment 1 — citation quality on the identical Sonnet-5 answer (480 test examples, `gemini-2.5-flash` judge):**

| Method | Recall | Precision | F1 | Cite length (tok) | Latency p50 | $/query |
|---|---|---|---|---|---|---|
| Prompted — gpt-5.5 add-citations *(post-hoc LLM)* | 0.840 | 0.885 | **0.851** | 35 | 8.3 s | $0.090 |
| **TokenPath** *(post-hoc attention)* | 0.735 | 0.938 | **0.815** | 114 | **1.60 s** | **$0.013** |
| Citations API — native *(Sonnet-5, generation-time)* | 0.761 | 0.915 | **0.812** | 177 | 9.9 s | $0.093 |
| Embedding retrieve+rerank *(post-hoc)* | 0.645 | 0.630 | 0.622 | 34 | ~1 s + index¹ | $0.0014 |

¹ Embedding is the only method with a separate one-time **indexing** cost (embed every document sentence, ~6 s, amortized across queries in real RAG). Per-query retrieval tunes to ~1 s on a colocated/GPU stack. TokenPath, Citations API, and prompted have no index — they read `(doc, answer)` in one pass.

Three things stand out:

**1. Post-hoc attention matches generation-time quality.** TokenPath (0.815) edges the generation-time Citations API (0.812) and trails only the prompted LLM (0.851) — by 0.036, all of it recall (0.735 vs 0.840). It clears the published LongCite-8B + SelfCite anchor (0.782, different generations, context only) and crushes naive retrieval (0.622). Its **precision (0.938) is the highest of any method**.

**2. It's dramatically cheaper and faster.** ~1.6 s vs ~8–10 s (~6× faster), $0.013 vs ~$0.09 (~7× cheaper). And unlike the LLM methods, TokenPath's cost is *model-independent* — a flat $1 per 1M attributed tokens regardless of who wrote the answer — because it works post-hoc on any model's output.

**3. It doesn't touch your answer.** The Citations API *regenerates*: its citations are clean, but it ships a different answer than the one you validated (and cites 177-token passages to do it). TokenPath, prompted, and embedding all cite the exact answer you already have.

### The cost/quality frontier

The story is a Pareto frontier with two points on it: the prompted LLM buys the last 0.036 of F1 with ~5× the latency and ~7× the cost, and TokenPath holds the knee. The Citations API costs the same as the prompted pass but lands slightly below TokenPath on F1 — while regenerating your answer. *(Chart data: `data/main_comparison.csv`.)*

### Per-dataset

TokenPath is strongest on extractive QA and weakest on abstractive summarization, as expected — see "the honest limitation" below. *(Chart data: `data/per_dataset_f1.csv`.)*

| Dataset | Prompted | Citations API | TokenPath | Embedding |
|---|---|---|---|---|
| gov_report (summary) | 0.972 | 0.947 | 0.897 | 0.843 |
| hotpotqa (multi-hop QA) | 0.811 | 0.732 | 0.811 | 0.657 |
| longbench-chat (abstractive) | 0.769 | 0.758 | 0.736 | 0.367 |
| multifieldqa_en (QA) | 0.854 | 0.807 | 0.806 | 0.705 |

---

## Getting there: tuning the attention aggregation

*(This section describes the original 2026-07-09 run, on the API's then-current Llama-3.1-8B reference model — the aggregation recipe it produced is unchanged and is what the 0.815 run above uses.)*

TokenPath's raw output is a token×token attention heatmap. Turning that into sentence citations is an aggregation problem, and the naive version left a lot on the table. Our first cut scored **F1 0.723**. Three changes to how we read the heatmap took it to **0.785** — closing ~70% of the then-gap to the Citations API — with no change to the API call, latency, or cost:

| Step | F1 (reported-avg) |
|---|---|
| Baseline (sum attention mass → sentences) | 0.722 |
| + row-normalize each answer token (one-token-one-vote) | 0.759 |
| + raise mass threshold to 0.30 | 0.784 |
| + merge adjacent supporting sentences into passages | **0.785** |

*(Chart data: `data/tokenpath_tuning_trajectory.csv`.)*

The wins were mostly **precision**: normalizing each answer token's contribution (so a few high-magnitude tokens can't dominate) and raising the mass bar dropped junk citations — which, as a bonus, *also* raised recall, because a cleaner set of cited sentences makes the judge's support call easier.

We found this fast and honestly: cache the raw heatmaps once (they don't depend on the aggregation), then every candidate config is free to compute — we only pay the judge. We tuned on a **145-example error set** (the examples the baseline scored worst on) for high signal, then **validated the winner on the full 480**. Rejected ideas — max-pooling, concentration filtering, length-normalization, relative thresholds — are in the sweep logs (`data/sweep_leaderboard_dev.json`, `data/sweep_leaderboard_full.json`).

---

## The honest limitations

We'd rather state these than have you find them.

**1. A prompted frontier LLM still wins on raw quality.** Post-hoc attention (0.815) trails the prompted add-citations pass (0.851) by 0.036, and the gap is entirely recall (0.735 vs 0.840). It concentrates on **abstractive/paraphrastic** statements: when the answer says *"appointed with Senate confirmation"* but the document says *"advice and consent of the Senate,"* an LLM reasons across the paraphrase; attention does surface matching and can miss it. This is why TokenPath is strongest on extractive QA (hotpotqa 0.81, tied with the prompted pass) and weakest on abstractive chat (longbench-chat 0.74). Some of this gap is likely a floor for pure attention attribution, not a tuning knob.

**2. TokenPath cites at sentence/passage granularity.** Its citations average 114 tokens (whole sentences, and merged passages) vs the prompted LLM's surgical 35-token quotes. It reliably points you to the right *place*; it doesn't pinpoint the exact clause. (The Citations API is coarser still, at 177 tokens.)

**3. The judge is an LLM.** We use one consistent cheap judge across all methods, so comparisons are internally fair, but absolute F1 depends on the judge. Published LongCite anchors used `gpt-4o-2024-05-13`; we report ours and treat published rows as context.

**4. LongBench-Cite documents are 2023-era.** A stretch goal is re-running on 2025–26 documents to confirm the numbers hold.

---

## What this means

If you need the single highest-quality citations and don't mind a second frontier LLM call per query, a prompted add-citations pass still wins by a few points of F1 — on recall. But if you want **good citations on any model's output, cheaply, fast, without regenerating the answer or standing up a retrieval index** — post-hoc attention attribution now matches generation-time citation quality and is far ahead on everything else.

---

## Reproducibility

Everything is pinned in `benchmarking/config.py` (models, judge, thresholds, aggregation config) and cached at every stage. The full harness, the tuned aggregation, and the raw per-example judge outputs are in the repo under `benchmarking/`.

- Models: generator + prompted `openai/gpt-5.5`; Citations API `claude-sonnet-5`; judge `google/gemini-2.5-flash`; embedding `text-embedding-3-large` + `BAAI/bge-reranker-base`.
- TokenPath aggregation: `{threshold: 0.30, row_norm: true, merge_adjacent: true}` — unchanged across the reference-model upgrade.
- TokenPath runs against the hosted API as served on the run date (2026-07-15), which reads a Qwen3.5-9B-based reference model. The 2026-07-09 run — the API's previous Llama-3.1-8B reference model, F1 0.785 — is preserved in `data/exp1_methods.json` (`tokenpath_tuned_llama31_archived`).
- Data for every chart in this post is in `benchmarking/blog/data/` (CSV + JSON).

---

## Appendix: charts to generate

Raw data lives in `benchmarking/blog/data/`. Suggested figures (regenerate in the site theme; the data is theme-agnostic):

1. **Headline F1 bar chart** — four methods, reported-avg F1. Source: `main_comparison.csv` (`f1_reported`). Highlight the TokenPath bar.
2. **Cost/quality frontier (the money chart)** — scatter of F1 (y) vs $/query (x, log scale); optionally a second panel F1 vs latency. Source: `main_comparison.csv`. TokenPath should read as the knee of the frontier.
3. **Per-dataset grouped bars** — F1 by dataset × method. Source: `per_dataset_f1.csv`.
4. **Tuning trajectory** — line/step chart, F1 0.722 → 0.785 across the four tuning steps. Source: `tokenpath_tuning_trajectory.csv`.
5. **Citation length** — bar chart of mean citation length (tokens) per method, to visualize the granularity tradeoff. Source: `main_comparison.csv` (`cite_len_glm4`).

Full per-method detail (recall/precision, example-mean vs reported-avg, per-dataset F1, cites/statement) is in `exp1_methods.json`.

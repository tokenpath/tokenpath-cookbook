# TokenPath quality benchmark

Reproducible harness behind the post **"Post-hoc attribution matches
generation-time citation — at a fraction of the latency, on any model's output."**

The thesis, in one sentence: *post-hoc, token-level attribution matches or beats
generation-time citation on established benchmarks, at a fraction of the latency —
and works on any model's output.* Everything here serves that sentence.

TokenPath does something no one else does directly — it attributes an answer to
its source using an open-source model's **attention map**, after generation, on
output from *any* model. That makes it hard to benchmark head-to-head, because
the alternatives all do a different thing. So instead of a rigged apples-to-apples
table, we measure post-hoc attribution against the **three ways people actually
get citations today**, on benchmarks we didn't write:

| # | Experiment | Benchmark | What it shows |
|---|---|---|---|
| 1 | Citation quality | [LongBench-Cite](https://github.com/THUDM/LongCite) | F1 vs. prompted / regenerated / retrieved citation, plus latency & $/query in the same table |
| 2 | Attribution-guided citation selection | [WebCode](https://exa.ai/blog/webcode) (Exa) | Citation precision for all returned results vs. one shared attribution-mass selection rule |

## What we compare against (Experiment 1)

We freeze **one** set of answers (a single generator, no citations) and attribute
those *same* answers four ways, so we compare attribution, not generators:

1. **TokenPath** — one post-hoc `/v1/attributions/heatmap` call; attribution mass
   → sentence citations, threshold tuned on the val split only.
2. **Prompted citation** — an LLM "add citations" pass over the frozen answer.
3. **Embedding retrieve+rerank** — the naive "just embed the answer and search the
   source" alternative (local sentence-transformers).
4. **Anthropic Citations API (reproduction mode)** — the one method that *regenerates*.
   We report its **answer-preservation rate** as a first-class column: regeneration
   means you no longer ship the answer you validated.

Published **LongCite-8B / SelfCite** rows appear for context, greyed and labeled
"different generations." We also support **re-judging** the anchor generations with
our own judge (`rejudge_anchors.py`) so at least one row is a true like-for-like.

## Reproducibility bar

- **Everything pinned** — models, judge, thresholds, dates — in [`config.py`](config.py).
- **The judge is a faithful port of LongCite's `auto_scorer.py`** ([`common/judge.py`](common/judge.py)),
  prompts verbatim, so our F1 sits on the same ruler as the published rows. One
  consistent judge scores every row (ours *and* re-judged anchors).
- **Val/test discipline** — TokenPath's mass threshold is tuned on **val** and
  reported on **test**; the split is deterministic and leak-free ([`load_data.py`](exp1_longbench_cite/load_data.py)).
- **Real latency and cost** — measured at every call site; OpenRouter's usage
  accounting gives actual $/query, not a price table.
- **Checkpointed** — every expensive stage caches to `results/`, so a run resumes
  instead of re-paying.

### Two honest caveats (stated in the post, not buried)

1. LongBench-Cite's underlying texts are 2023-era. We note it; a stretch goal is
   re-running the protocol on 2025–26 docs to show the numbers hold.
2. In the WebCode comparison, attribution runs after answer generation while
   retrieval happens upstream. Exp 2 measures citation selection among results
   already returned; it does not compare retrieval quality.

## Setup

```bash
pip install -r benchmarking/requirements.txt
# Optional (embedding baseline + GLM-4 citation length):
#   pip install sentence-transformers transformers

export TOKENPATH_API_KEY=tpk_...        # https://platform.tokenpath.ai
export TP_TOKENPATH_BACKEND_ID=qwen-... # optional cache/provenance tag; change with backend
export OPENROUTER_API_KEY=sk-or-...      # generator + prompted baseline + judge
export ANTHROPIC_API_KEY=sk-ant-...      # optional: Citations API baseline only
```

Secrets are read from the environment only and are **never** written to the repo.

## Run it

```bash
# Smoke test the whole pipeline on a tiny slice first (cheap):
SMOKE=1 bash benchmarking/scripts/run_all.sh

# Full run:
bash benchmarking/scripts/run_all.sh
```

Or one table at a time:

```bash
# Exp 1
python -m benchmarking.exp1_longbench_cite.load_data --download
python -m benchmarking.exp1_longbench_cite.tune_threshold --limit-per-dataset 10
python -m benchmarking.exp1_longbench_cite.run --split test
python -m benchmarking.exp1_longbench_cite.make_table

# Exp 2 (assemble data/webcode/webcode.jsonl first — schema in exp2_webcode/load_data.py)
python -m benchmarking.exp2_webcode.load_data --make-sample   # offline fixture
python -m benchmarking.exp2_webcode.run --data benchmarking/data/webcode/sample.jsonl
python -m benchmarking.exp2_webcode.make_chart

```

## Data you supply

- **LongBench-Cite** downloads automatically from THUDM/LongCite (1000 examples;
  we default to the English subset for the post — flip `--all-languages` for the
  full published set).
- **WebCode** is not redistributed here. Assemble one JSONL from Exa's release plus
  your own provider API calls, one line per `(query, provider)` — the exact schema
  and a synthetic fixture are in [`exp2_webcode/load_data.py`](exp2_webcode/load_data.py).
- **Anchor generations** (LongCite-8B / SelfCite) for re-judging come from their
  repos; we don't redistribute them.

## The one knob that matters: the judge

`config.JUDGE_MODEL` defaults to a **cheap** OpenRouter model, used identically for
our methods and for re-judged anchors — internally consistent, so the F1 column is
apples-to-apples. The published anchor rows were judged with `gpt-4o-2024-05-13`.
For exact parity with those numbers, set:

```bash
export TP_JUDGE_MODEL=openai/gpt-4o-2024-05-13
```

and re-run. We report both our-judge numbers and (greyed) published numbers so the
judge difference is never hidden.

## Layout

```
benchmarking/
  config.py                     pinned models, thresholds, dates, pricing
  common/                       tokenpath client, openrouter client, judge (LongCite port),
                                segmentation, timing/cost, citation-length, io/caching
  exp1_longbench_cite/          loader, frozen-answer gen, 4 methods, tune, re-judge, run, table
  exp2_webcode/                 loader, generate→attribute→select→score, run, chart
  scripts/run_all.sh            one command per table
  results/                      tables (.md), figures (.png), summary scores (.json)
```

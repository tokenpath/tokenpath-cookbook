# Handoff — TokenPath quality benchmark (blog 1)

Pick-up doc for the next agent/engineer. Read this top to bottom before running
anything.

**Branch:** `claude/tokenpath-benchmark-blog-4xuuwx` (all code + this doc live here).
**Repo:** `tokenpath/tokenpath-cookbook`, harness under `benchmarking/`.

---

## 1. Goal (don't lose the thesis)

One post, one claim:

> *Post-hoc, token-level attribution matches or beats generation-time citation on
> established benchmarks, at a fraction of the latency — and works on any model's
> output.*

TokenPath = post-hoc attribution from an open-source model's **attention map**,
after generation, on output from **any** model. Nobody does this directly, so we
benchmark it against the three ways people get citations today: **prompted**,
**regenerated**, **retrieved**. Everything in the harness serves that sentence.

Full plan is in the original task description; the harness implements it. See
`benchmarking/README.md` for the reproducibility story.

---

## 2. Current state

**Done — the entire harness is written, compiles, and all non-network logic is
validated offline.** Two experiments:

- **Exp 1 (LongBench-Cite)** — `benchmarking/exp1_longbench_cite/`
  Freeze one answer set (single generator, no citations), attribute it 4 ways
  (TokenPath, prompted, embedding retrieve+rerank, Anthropic Citations API
  reproduction mode), judge with a **verbatim port of LongCite's `auto_scorer.py`**
  routed through OpenRouter. Table has F1/R/P + citation length + latency p50 +
  $/query + answer-preservation. Val/test threshold discipline + anchor re-judging.
- **Exp 2 (WebCode)** — `benchmarking/exp2_webcode/`
  Generate an answer from each provider's results, attribute it, select results by
  attribution mass under ONE shared threshold, and compare citation precision for
  all returned results with the selected set. Chart + table.

**What was validated offline** (no API): segmentation, judge `[[...]]` parsing,
heatmap→mass math, split determinism + leak-freeness, precision aggregation,
and table/chart rendering. All Python modules import and `py_compile` clean.

**What was NOT run:** anything touching a live API (see blocker below). So the
API *response shapes* are implemented from the cookbook notebooks + public docs
and still need one live smoke test to confirm. See §6.

---

## 3. THE BLOCKER — network egress (do this first)

This session's environment blocks the domains the harness needs. Confirmed via
the agent proxy (`connect_rejected`, policy 403):

| Domain | Needed for | Status at handoff |
|---|---|---|
| `api.tokenpath.ai` | all experiments | ❌ blocked |
| `openrouter.ai` | generator, prompted baseline, judge | ❌ blocked |
| `huggingface.co` | embedding baseline models + GLM-4 cite-length tokenizer | ❌ blocked |
| `api.exa.ai` (+ other provider APIs) | Exp 2 data assembly | ❌ blocked |
| `api.anthropic.com` | Citations API baseline (optional) | ✅ reachable |
| `github.com`, `raw.githubusercontent.com`, `pypi.org` | data + deps | ✅ reachable |

**Action:** widen the environment's network policy to allow the blocked hosts
(Claude Code environment settings → network policy;
docs: https://code.claude.com/docs/en/claude-code-on-the-web). Re-check with:

```bash
for u in https://api.tokenpath.ai https://openrouter.ai https://huggingface.co; do
  printf "%-28s " "$u"; curl -sS -o /dev/null -w "%{http_code}\n" --max-time 10 "$u"; done
```

`000` = still blocked; a real HTTP code = reachable.

---

## 4. Keys

Env vars only — **never** commit them (`benchmarking/.gitignore` covers `*.env`).

```bash
export TOKENPATH_API_KEY=tpk_...       # provided by user (ROTATE — shared in chat)
export OPENROUTER_API_KEY=sk-or-...    # provided by user (ROTATE — shared in chat)
export ANTHROPIC_API_KEY=sk-ant-...    # NOT yet provided — needed only for Exp 1 Citations baseline
```

> ⚠️ The TokenPath + OpenRouter keys were pasted into the chat transcript. Ask the
> user to rotate them once benchmarking is done.

The user asked to **use OpenRouter for all LLM access and cheap models**.
`config.py` defaults to `openai/gpt-4o-mini` for generator/prompted/judge.

---

## 5. How to run (once egress + keys are set)

```bash
pip install -r benchmarking/requirements.txt
# optional (embedding baseline + GLM-4 cite length):
#   pip install sentence-transformers transformers

# 1. SMOKE TEST FIRST — tiny slice, validates every API shape cheaply:
SMOKE=1 bash benchmarking/scripts/run_all.sh

# 2. Full run:
bash benchmarking/scripts/run_all.sh
```

Per-experiment commands are in `benchmarking/README.md` and `scripts/run_all.sh`.
Outputs land in `benchmarking/results/` (tables `.md`, figures `.png`, summary
`.json`). Every stage is cached/resumable.

---

## 6. Assumptions to verify against the live API (IMPORTANT)

These were implemented from the cookbook notebooks + public docs but never hit a
live endpoint. **The smoke test will surface any mismatch.** Check each:

1. **`/v1/attributions/heatmap` response shape** — code expects
   `{shape:[a,d], row:[], col:[], data:[], answer_offsets:[[s,e]...],
   document_offsets:[[s,e]...]}` (from `notebooks/heatmap-visualization.ipynb`).
   Verified against the notebook; confirm live. Parsed in
   `common/tokenpath.py::Heatmap.from_response`.
2. **TokenPath cost model** — we approximate attributed tokens as
   `len(answer_offsets)+len(document_offsets)` × $1/1M
   (`common/timing.py::tokenpath_cost_usd`). Confirm what TokenPath actually bills
   (attributed tokens definition) and fix if needed — this drives the $/query
   column for the TokenPath row.
3. **OpenRouter `usage.cost`** — we send `usage:{include:true}` and read
   `usage.cost` for real $/query (`common/openrouter.py`). Confirm the field is
   populated for the chosen models; if not, add a price table fallback.
4. **Anthropic Citations API response shape** — we read `content[].citations[]`
   with `cited_text` char spans (`methods/citations_api_method.py`). Confirm block
   structure + that `cited_text` is verbatim-in-document. Needs `ANTHROPIC_API_KEY`.
5. **Heatmap latency** — the whole latency story rests on the heatmap being ONE
   fast post-hoc call. Confirm p50 on real docs (LongBench contexts are long).

If a heatmap on a 128k-token LongBench context is slow or the API prefers
`/v1/attributions` (span→source) over the heatmap, note that
`methods/tokenpath_method.py` currently uses the heatmap for richer multi-span
citations (better recall). A `/v1/attributions`-based variant is a small rewrite
if needed.

---

## 7. Pending decisions (raised with user, not yet answered)

1. **Anthropic Citations API baseline** — needs `ANTHROPIC_API_KEY` (proprietary
   feature, can't go through OpenRouter). Gated/skipped without one. Decide: get a
   key, or drop this baseline (lose the answer-preservation finding).
2. **WebCode data (Exp 2)** — not public-redistributable. Must be assembled from
   Exa's WebCode release + the 5 providers' search APIs (Exa, Brave, Perplexity,
   Parallel, Tavily — each needs a key). Loader schema + synthetic fixture in
   `exp2_webcode/load_data.py`. Decide: obtain provider access, or defer Exp 2.
3. **Judge parity** — default judge is cheap (`openai/gpt-4o-mini`), consistent
   across all rows. Published anchors used `gpt-4o-2024-05-13`. For exact parity
   set `TP_JUDGE_MODEL=openai/gpt-4o-2024-05-13`. Decide whether the post needs
   exact parity or the "one consistent cheap judge" framing is enough (we report
   both, published rows greyed).

---

## 8. Gotchas / design choices baked in

- **English subset by default** — Exp 1 defaults to `{longbench-chat, hotpotqa,
  gov_report, multifieldqa_en}` (blog is English-facing). `--all-languages` adds
  the Chinese sets for the full published average. Note: `eval_cite.py` excludes
  `multifieldqa_en/zh` from its headline average; we match that (`avg_reported`)
  AND report `avg_all_datasets`.
- **Threshold is a placeholder until tuned** — `config.TOKENPATH_MASS_THRESHOLD`
  (0.15) is overwritten by `tune_threshold.py` → `results/exp1_threshold.json`,
  which `run.py` reads. Always run tuning before the test run.
- **Frozen answers are the fairness cornerstone** — all methods attribute the SAME
  `results/exp1_frozen_answers.jsonl` except Citations API (which regenerates, by
  design — that asymmetry is the answer-preservation column, not a bug).
- **Citation length** uses the GLM-4 tokenizer to match LongCite; falls back to
  word count (labeled) if `transformers` isn't installed.
- **Anchor re-judging** (`rejudge_anchors.py`) needs the anchor models' actual
  generation files in LongCite schema (from THUDM/LongCite `preds/` and
  facebookresearch/SelfCite) — we don't redistribute them. Without them, use the
  greyed published row in `make_table.py::PUBLISHED`.

---

## 9. File map

```
benchmarking/
  config.py                 pins: models, judge, thresholds, dates, pricing, RunConfig
  common/
    tokenpath.py            client (attribute + heatmap) + Heatmap mass helpers
    openrouter.py           chat client, real cost via usage accounting, retries
    judge.py                LongCite auto_scorer port (verbatim prompts) via OpenRouter
    segment.py              shared sentence/statement segmentation
    cite_len.py             GLM-4 citation length (guarded)
    timing.py               latency p50/p95 + $/query aggregation
    io_utils.py             jsonl cache, dedup, json read/write
    env.py                  build clients from env vars
  exp1_longbench_cite/
    load_data.py            download LongBench-Cite, deterministic val/test split
    freeze_answers.py       generate one no-citation answer per example
    methods/                base + tokenpath / prompted / embedding / citations_api
    tune_threshold.py       tune TokenPath mass threshold on VAL -> exp1_threshold.json
    run.py                  cite -> judge -> aggregate, per method
    rejudge_anchors.py      re-score anchor generations with our judge
    make_table.py           render results table (+ published + re-judged rows)
  exp2_webcode/
    load_data.py            WebCode loader + schema + --make-sample fixture
    select_and_score.py     generate -> ground -> attribute -> select -> precision
    run.py / make_chart.py  orchestrate + chart/table
  scripts/run_all.sh        one command per table (SMOKE=1 for a tiny slice)
  results/                  outputs (summaries/tables/figures committed; caches gitignored)
  README.md                 repro doc
  HANDOFF.md                this file
```

---

## 10. Suggested order of work for next agent

1. Open egress (§3), export keys (§4).
2. `SMOKE=1 bash benchmarking/scripts/run_all.sh` — fix any API-shape mismatch (§6).
3. Decide pending items (§7) with the user.
4. `tune_threshold` on val, then full Exp 1 on test; sanity-check numbers against
   the published anchors (LongCite-8B+SelfCite ≈ F1 0.78, GPT-4o judge).
5. Assemble WebCode data → Exp 2.
6. Draft the post from the tables/figures (structure in the plan / README).
7. Open a PR (user hasn't asked for one yet — confirm first).

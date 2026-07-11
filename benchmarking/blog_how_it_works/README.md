# Blog post materials — "How TokenPath works"

The internals/explainer post, written **cookbook-first**: a short intro, then
mostly runnable code with real outputs — turn the attention heatmap into a
citation, three things string matching can't do (cross-lingual, paraphrase,
tables), an optional raw-Transformers reproduction on Llama-3.1-8B, and the
benchmark + measurement-scope wrap-up.

## Files
- **`post.md`** — the blog post (Markdown), cookbook-style. Every code block's
  output is a real captured run.
- **`../../notebooks/how-tokenpath-works.ipynb`** — the runnable companion
  notebook (mostly mirrors the post; not 1:1). **This is the code we ship.** It's
  executed with outputs baked in, including the heatmap figure. The API sections
  run on just a `TOKENPATH_API_KEY`; the local-model section (§3) is behind a
  `RUN_LOCAL` guard (ships `False`, so top-to-bottom runs light; outputs shown are
  from a real `RUN_LOCAL=True` run).
- **`reproduce_attention.py`** / **`find_attribution_heads.py`** — the standalone
  minimal scripts the post's §3 quotes. Dependency-light versions of the
  notebook's local-model section; handy to run outside a notebook. NOT the
  production implementation (O(n²), single-doc; the post's §3.3 explains why).

## Running
- **Notebook (recommended):** open in Jupyter/Colab, set `TOKENPATH_API_KEY`, run
  all. Flip `RUN_LOCAL = True` to also run the local 8B reproduction.
- **Scripts:** `pip install "transformers>=4.44" torch accelerate`, then
  `python reproduce_attention.py`. Defaults to the gated
  `meta-llama/Llama-3.1-8B-Instruct`; set `LLAMA_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct`
  (identical weights, ungated) if you lack meta-llama access. GPU (bf16)
  recommended; the tiny demo doc runs on CPU.

## Known limitation (tracked)
The §2.4 "honest note on repeated values" refers to a real failure mode:
first-occurrence bias when the same value appears in parallel roles and
disambiguation must come from the *question* (the winner doesn't flip; only
confidence drops). Full repro + hypotheses are in **tokenpath/tokenpath#149**
(private). Do **not** link that private issue from the public post. The §2.1 date
example disambiguates fine because each claim carries its own context in the
*answer* — a different, working case.

## Notes for publishing
- **Every output in `post.md` and the notebook is a real captured run.** API
  outputs are live TokenPath calls; §3 outputs are real Llama-3.1-8B-Instruct runs
  (captured on CPU, seed 0). A GPU bf16 run may differ in the last decimal but not
  the ranking. Note: the single-best-span confidence for the Oregon "green and
  yellow" example is ~0.7–0.9 across runs (mild nondeterminism); the sentence
  ranking is stable.
- The head layer/head numbers are a property of Llama-3.1-8B-Instruct on synthetic
  probes, **not** a leak of the hosted API's selection. The post and notebook say
  so explicitly. Keep it that way.
- The post deliberately does **not** disclose the production head selection or the
  long-context attention kernel; it explains the *principle* (thin answer×doc
  slice, selected heads, tiled recompute à la FlashAttention) and points at the
  FlashAttention paper. Confidence without IP leak was the explicit goal.
- Benchmark F1 figures must stay in sync with `../blog/post.md`: TokenPath 0.785,
  LLM methods 0.81–0.85, retrieval 0.62.

## Suggested figures (for the site)
1. **The attention heatmap** — already rendered in the notebook (answer tokens ×
   document tokens for the Oregon Duck example). Regenerate in the site theme from
   the notebook's plotting cell.
2. **Head leaderboard** — bar chart of top attribution heads by needle-mass vs the
   chance baseline. Source: the §3.2 probe output.
3. **Long-context schematic** — full seq×seq matrix (greyed, "never materialized")
   vs the thin answer-rows × doc-cols slice we recompute. For §3.3.

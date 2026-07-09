# Experiment 1 — citation quality on the SAME answer (final)

Every method is judged on the **identical** naturally-generated Sonnet-5 answer
(480 LongBench-Cite test examples; Gemini-2.5-flash judge; reported-avg F1, the
LongCite headline convention). This isolates attribution QUALITY — same
statements, different citations — and each method makes **one call per
(document, question, answer) triple**.

| Method | Recall | Precision | F1 | Cite len (glm4) | cites/stmt | Latency p50 | $/query |
|---|---|---|---|---|---|---|---|
| Prompted — gpt-5.5 add-citations *(post-hoc LLM)* | 0.840 | 0.885 | **0.851** | 35 | 1.04 | 8.3s | $0.090 |
| Citations API — native *(Sonnet-5, generation-time)* | 0.761 | 0.915 | **0.812** | 177 | 0.67 | 9.9s | $0.093 |
| **TokenPath — tuned** *(post-hoc attention)* | 0.712 | 0.898 | **0.785** | 106 | 0.79 | **1.68s** | **$0.0125** |
| Embedding retrieve+rerank *(post-hoc)* | 0.645 | 0.630 | 0.622 | 34 | 2.01 | ~1s query + ~6s index¹ | $0.0014 |

¹ **Embedding latency — honest accounting.** Embedding is the only method with a two-part latency: a one-time **indexing** cost (embed every document sentence, ~6s, amortized across all queries in real RAG — you build the index once) and a per-query **retrieval** cost. The retrieval *work* is light — embed ~30 answer sentences + cosine search + rerank ~150 short pairs — and tunes to **~1s (or less) on a colocated/GPU stack**. The multi-second figures we measured in-harness were inflated by a CPU reranker and per-call hosted-API network round-trips (~1.5s each; `text-embedding-3-small` and `-large` timed identically, so it's round-trip, not model). TokenPath / CA / prompted have **no index** — they read (doc, answer) in one call — so there's nothing to amortize and no separate query/index split.

**Reading it**
- **Quality:** post-hoc LLM (prompted, 0.851) ≥ generation-time (Citations API, 0.812) > **TokenPath (0.785)** ≫ embedding (0.622). TokenPath's precision (0.898) is right with the LLM methods.
- **Latency / cost:** TokenPath is **~5–8× faster and ~7× cheaper** than the two LLM-based methods, and works post-hoc on *any* model's output.
- **No index:** embedding has a structural cost the others don't — it must embed every document sentence (**~6.2s one-time indexing**), then per query embed the answer + search + rerank (**~5.6s**, inflated by a CPU reranker; sub-second on GPU/hosted). TokenPath / CA / prompted read (doc, answer) in one pass — nothing to index.

**Models**
- Prompted & generator: `openai/gpt-5.5`; Citations API: `claude-sonnet-5`; judge: `google/gemini-2.5-flash`.
- Embedding: hosted `text-embedding-3-large` retrieval (top-5) + local `BAAI/bge-reranker-base` rerank (sigmoid ≥ 0.70). Retrieval-only (no rerank) scores F1 0.518 — the reranker lifts precision 0.45 → 0.63. 16/480 docs errored on embedding (≈3%, empty citations, slightly depresses its score). A hosted reranker (needs a production key) would replace the local CPU one.

**TokenPath tuning (this run):** aggregation `{threshold 0.30, row_norm, merge_adjacent}` (`config.TOKENPATH_AGG`), which took TokenPath from 0.723 → 0.785 reported-avg (0.754 → 0.810 example-mean) — see the sweep in `results/sweep/`.

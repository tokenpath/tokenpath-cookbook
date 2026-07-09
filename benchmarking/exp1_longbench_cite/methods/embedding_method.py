"""Embedding retrieve(+rerank) baseline — hosted OpenAI embeddings + local reranker.

The naive alternative from our docs: for each answer sentence, embed it and the
document sentences, retrieve the top-k document sentences by cosine similarity,
rerank them with a cross-encoder, and keep those above a threshold as citations.
"Retrieval, but at citation time" — the honest strawman for 'why not just embed
the answer and search the source?'

Retriever is hosted (OpenAI text-embedding-3-large); reranker is a local modern
cross-encoder (BAAI/bge-reranker-base). Set RERANK_MODEL="" for retrieval-only.

Two latency numbers are recorded, because retrieval has a structural cost the
other methods don't: **indexing** the document (embedding all its sentences) is a
one-time, answer-independent cost you'd precompute and reuse in real RAG;
**retrieval** (embed the answer's sentences, search, rerank) is what you pay per
query. TokenPath / Citations API / prompted have no index — they read the
(doc, answer) fresh every call — so the fair per-query latency for embedding is
the retrieval part, with indexing reported separately.
"""

from __future__ import annotations

import os
import time

import numpy as np
import requests

from ... import config
from ...common.segment import statement_spans
from ...common.segment import statements as segment_statements
from .base import CitedAnswer, Method

OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
_reranker = None


def _load_reranker(model: str):
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # type: ignore
        _reranker = CrossEncoder(model)
    return _reranker


def _embed(texts: list[str], model: str, api_key: str, timeout: int = 120,
           max_retries: int = 6) -> tuple[np.ndarray, int]:
    if not texts:
        return np.zeros((0, 1)), 0
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENAI_EMBED_URL,
                headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
                json={"model": model, "input": texts}, timeout=timeout,
            )
        except requests.RequestException as exc:
            last = exc; time.sleep(min(2 ** attempt, 30)); continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last = RuntimeError(f"openai embeddings {resp.status_code}: {resp.text[:200]}")
            ra = resp.headers.get("retry-after")
            time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 30))
            continue
        resp.raise_for_status()
        data = resp.json()
        mat = np.array([d["embedding"] for d in data["data"]], dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms > 0, norms, 1.0), int(data.get("usage", {}).get("total_tokens", 0))
    raise last


class EmbeddingMethod(Method):
    name = "embedding"

    def __init__(
        self,
        embed_model: str = config.EMBED_MODEL,
        rerank_model: str | None = None,
        top_k: int = config.EMBED_TOP_K,
        score_threshold: float = config.EMBED_SCORE_THRESHOLD,
        api_key: str | None = None,
    ):
        self.embed_model = embed_model
        self.rerank_model = (config.RERANK_MODEL if rerank_model is None else rerank_model) or None
        self.rerank_sigmoid = getattr(config, "RERANK_SIGMOID", True)
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def cite(self, example: dict, answer: str) -> CitedAnswer:
        document, query = example["context"], example["query"]
        doc_sents = [document[s:e] for s, e in statement_spans(document)]
        segs = segment_statements(answer)
        st_texts = [s["statement"] for s in segs]

        # INDEX phase (amortizable): embed the document's sentences.
        ti = time.perf_counter()
        doc_emb, dtok = _embed(doc_sents, self.embed_model, self.api_key)
        index_latency = time.perf_counter() - ti

        # RETRIEVAL phase (per query): embed answer sentences, search, rerank.
        tr = time.perf_counter()
        st_emb, stok = _embed(st_texts, self.embed_model, self.api_key)
        sims = st_emb @ doc_emb.T if doc_emb.size and st_emb.size else np.zeros((len(segs), 0))
        tops = [np.argsort(sims[i])[::-1][: self.top_k] if sims.shape[1] else np.array([], int)
                for i in range(len(segs))]

        reranker = _load_reranker(self.rerank_model) if self.rerank_model else None
        if reranker is not None:
            pairs, owner = [], []
            for i, top in enumerate(tops):
                for j in top:
                    pairs.append((st_texts[i], doc_sents[int(j)])); owner.append((i, int(j)))
            scores = reranker.predict(pairs, batch_size=64) if pairs else []
            if self.rerank_sigmoid and len(scores):
                scores = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))
            kept: dict[int, list[dict]] = {i: [] for i in range(len(segs))}
            for (i, j), sc in zip(owner, scores):
                if sc >= self.score_threshold:
                    kept[i].append({"cite": doc_sents[j], "score": round(float(sc), 4)})
            statements = [{"statement": st_texts[i], "span": segs[i]["span"], "citation": kept[i]}
                          for i in range(len(segs))]
        else:  # retrieval-only: cosine threshold
            statements = []
            for i, seg in enumerate(segs):
                c = [{"cite": doc_sents[int(j)], "score": round(float(sims[i][j]), 4)}
                     for j in tops[i] if sims[i][j] >= self.score_threshold]
                statements.append({"statement": st_texts[i], "span": seg["span"], "citation": c})
        retrieval_latency = time.perf_counter() - tr

        cost = (dtok + stok) / 1_000_000 * config.EMBED_USD_PER_MTOK
        return CitedAnswer(
            idx=example["idx"], dataset=example["dataset"], query=query,
            prediction=answer, statements=statements, method=self.name,
            latency_s=retrieval_latency,  # fair per-query latency (index is amortized)
            cost_usd=cost,
            extra={"embed_model": self.embed_model, "rerank_model": self.rerank_model,
                   "top_k": self.top_k, "score_threshold": self.score_threshold,
                   "index_latency_s": round(index_latency, 3),
                   "retrieval_latency_s": round(retrieval_latency, 3),
                   "embed_tokens": dtok + stok, "hosted_embed": True},
        )

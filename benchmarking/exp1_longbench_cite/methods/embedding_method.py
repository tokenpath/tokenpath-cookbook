"""Embedding retrieve+rerank baseline — the naive alternative from our docs.

For each answer sentence, embed it, retrieve the top-k document sentences by
cosine similarity, rerank them with a cross-encoder, and keep those above a
threshold as citations. This is "retrieval, but at citation time" — the honest
strawman for 'why not just embed the answer and search the source?'

Runs locally via sentence-transformers (no third-party embedding key). The model
downloads once from HuggingFace; the import is guarded so the rest of the harness
runs even where the models aren't installed. Latency is measured locally; dollar
cost is 0 (self-hosted) — we note that in the table rather than pretending it is
free of compute.
"""

from __future__ import annotations

import time

from ... import config
from ...common.segment import statement_spans
from ...common.segment import statements as segment_statements
from .base import CitedAnswer, Method

_embedder = None
_reranker = None


def _load(embed_model: str, rerank_model: str):
    global _embedder, _reranker
    if _embedder is None:
        from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore

        _embedder = SentenceTransformer(embed_model)
        _reranker = CrossEncoder(rerank_model)
    return _embedder, _reranker


class EmbeddingMethod(Method):
    name = "embedding"

    def __init__(
        self,
        embed_model: str = config.EMBED_MODEL,
        rerank_model: str = config.RERANK_MODEL,
        top_k: int = config.EMBED_TOP_K,
        score_threshold: float = config.EMBED_SCORE_THRESHOLD,
    ):
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.top_k = top_k
        self.score_threshold = score_threshold

    def cite(self, example: dict, answer: str) -> CitedAnswer:
        import numpy as np

        embedder, reranker = _load(self.embed_model, self.rerank_model)
        document, query = example["context"], example["query"]

        doc_spans = statement_spans(document)
        doc_sents = [document[s:e] for s, e in doc_spans]
        segs = segment_statements(answer)

        t0 = time.perf_counter()
        doc_emb = embedder.encode(doc_sents, normalize_embeddings=True, show_progress_bar=False)
        statements = []
        for seg in segs:
            st_text = seg["statement"]
            q_emb = embedder.encode([st_text], normalize_embeddings=True, show_progress_bar=False)[0]
            sims = doc_emb @ q_emb
            top = np.argsort(sims)[::-1][: self.top_k]
            pairs = [(st_text, doc_sents[i]) for i in top]
            rerank_scores = reranker.predict(pairs) if pairs else []
            kept = [
                {"cite": doc_sents[i], "rerank_score": float(score)}
                for i, score in zip(top, rerank_scores)
                if score >= self.score_threshold
            ]
            statements.append(
                {"statement": st_text, "span": seg["span"], "citation": kept}
            )
        latency = time.perf_counter() - t0

        return CitedAnswer(
            idx=example["idx"],
            dataset=example["dataset"],
            query=query,
            prediction=answer,
            statements=statements,
            method=self.name,
            latency_s=latency,
            cost_usd=0.0,  # self-hosted; compute cost not billed
            extra={
                "embed_model": self.embed_model,
                "rerank_model": self.rerank_model,
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "self_hosted": True,
            },
        )

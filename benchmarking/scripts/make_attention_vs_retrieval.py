"""Build blog/data/attention_vs_retrieval_examples.json from judged Exp-1 runs.

A candidate is a statement where TokenPath's citation was judged FULLY
supported (support_score == 1) while embedding retrieve+rerank's was not
(support_score < 1), with both methods having actually cited something. These
are the "attention beats retrieval" cases the blog post counts and features.

Run from benchmarking/:  python scripts/make_attention_vs_retrieval.py
Inputs:  results/exp1nat_judged_tokenpath.jsonl, results/exp1nat_judged_embedding.jsonl
Output:  blog/data/attention_vs_retrieval_examples.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENPATH_JUDGED = ROOT / "results" / "exp1nat_judged_tokenpath.jsonl"
EMBEDDING_JUDGED = ROOT / "results" / "exp1nat_judged_embedding.jsonl"
OUT = ROOT / "blog" / "data" / "attention_vs_retrieval_examples.json"

# The three examples featured in the blog post, keyed by (idx, statement prefix).
FEATURED_KEYS = [
    (568, "The Oregon Duck mascot wears"),
    (597, "Rønnaug Alten made her stage debut"),
    (505, "So while Henry Roth was born"),
]


def cite_text(c):
    return c["cite"] if isinstance(c, dict) else c


def collect(path):
    out = {}
    for line in path.open():
        rec = json.loads(line)
        for s in rec["statements"]:
            out[(rec["idx"], s["statement"])] = (rec, s)
    return out


def main():
    tokenpath = collect(TOKENPATH_JUDGED)
    embedding = collect(EMBEDDING_JUDGED)

    candidates = []
    for key, (rec, tp) in sorted(tokenpath.items()):
        hit = embedding.get(key)
        if hit is None:
            continue
        emb = hit[1]
        if not tp.get("citation") or not emb.get("citation"):
            continue
        if tp.get("support_score") != 1:
            continue
        emb_support = emb.get("support_score")
        if emb_support is None or emb_support >= 1:
            continue
        candidates.append({
            "idx": rec["idx"],
            "dataset": rec["dataset"],
            "query": rec["query"],
            "statement": tp["statement"],
            "tokenpath_citation": [cite_text(c) for c in tp["citation"]],
            "tokenpath_support": tp["support_score"],
            "embedding_citation": [cite_text(c) for c in emb["citation"]],
            "embedding_support": emb_support,
        })

    featured = []
    for idx, prefix in FEATURED_KEYS:
        match = [c for c in candidates
                 if c["idx"] == idx and c["statement"].startswith(prefix)]
        if not match:
            raise SystemExit(f"featured example no longer qualifies: {idx} {prefix!r}")
        featured.extend(match)

    OUT.write_text(json.dumps({
        "featured": featured,
        "count_total": len(candidates),
        "all_candidates": candidates,
    }, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(candidates)} candidates ({len(featured)} featured) -> {OUT}")


if __name__ == "__main__":
    main()

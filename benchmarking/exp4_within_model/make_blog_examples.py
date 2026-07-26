"""Extract per-statement qualitative examples for the Exp 4 blog post.

Pairs every scored statement across the two conditions that cite the SAME frozen
answer — `prompted_thinking` (the arm the post quotes) and `tokenpath` — and emits
the cases where one was judged fully supported and the other was not.

Two things this is strict about, because the post quotes these examples verbatim:

  - Statements are paired only when their span AND text match exactly across the
    two conditions. Both segment the same frozen answer with the same segmenter,
    so they should align 1:1; any pair that does not is dropped and counted, never
    guessed at.
  - "Fully supported" means the judge returned [[Fully supported]] (support_score
    1.0) for a statement that actually carried a citation. A statement with no
    citation also scores 1.0 when the judge decides it needed none — that is not
    a supported citation and is excluded from both directions.

Counts are emitted in BOTH directions, including the one where asking the model
beat its attention.
"""

from __future__ import annotations

import argparse
import json
import os

from ..exp1_longbench_cite import load_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "blog", "data",
                        "exp4_examples.json")


def _latest(path: str) -> dict:
    out: dict = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["idx"]] = rec
    return out


def _cites(statement: dict) -> list[str]:
    return [c["cite"] for c in statement.get("citation", [])]


def _supported(statement: dict) -> bool:
    """Judged fully supported AND actually carrying a citation."""
    return statement.get("support_score") == 1 and bool(statement.get("citation"))


def _is_fragment(text: str) -> bool:
    """Segmentation debris, not a citable statement.

    The shared sentence segmenter splits on `[.!?]`, so abbreviations fracture:
    "Lt. Col. James Wilkerson" yields statements "Lt.", "Col." and so on. These
    are 4% of judged statements and they bias the per-statement comparison toward
    attention, which trivially "supports" the fragment "v." by citing "v." back
    while the prompted model quotes something meaningful and is marked unsupported.
    Fragments are 9% of attention's per-statement wins against 2% of asking's.

    Excluded from the counts because "901 statements" should mean 901 real
    statements. Note this does NOT flatter attention in aggregate: removing
    fragments entirely moves attention's F1 from 0.738 to 0.751 and asking's from
    0.603 to 0.602, i.e. the reported gap is slightly conservative.
    """
    return len(text) < 25 or len(text.split()) < 4


def build(split: str, seed: int, n_featured: int) -> dict:
    only = {e["idx"] for e in load_data.load_split(split, seed=seed, english_only=True)}
    examples = {e["idx"]: e for e in load_data.load_split(split, seed=seed,
                                                          english_only=True)}
    asked = _latest(os.path.join(RESULTS_DIR, "exp4_judged_prompted_thinking.jsonl"))
    attn = _latest(os.path.join(RESULTS_DIR, "exp4_judged_tokenpath.jsonl"))

    attn_better: list[dict] = []
    asked_better: list[dict] = []
    n_pairs = n_misaligned = n_fragments = 0

    for idx in sorted(only & set(asked) & set(attn)):
        a_stmts, t_stmts = asked[idx]["statements"], attn[idx]["statements"]
        for a, t in zip(a_stmts, t_stmts):
            # Never pair statements we cannot prove are the same statement.
            if a.get("span") != t.get("span") or a.get("statement") != t.get("statement"):
                n_misaligned += 1
                continue
            # The judge only scores the first max_statement_num statements; the
            # rest carry no verdict and cannot be compared.
            if a.get("support_score") is None or t.get("support_score") is None:
                continue
            n_pairs += 1
            if _is_fragment(a["statement"]):
                n_fragments += 1
                continue

            row = {
                "idx": idx,
                "dataset": asked[idx]["dataset"],
                "query": examples[idx]["query"],
                "statement": a["statement"],
                "tokenpath_citation": _cites(t),
                "tokenpath_support": t.get("support_score"),
                "asked_citation": _cites(a),
                "asked_support": a.get("support_score"),
            }
            # The loser must actually have been penalised. A statement with no
            # citation scores 1.0 when the judge rules none was needed — counting
            # that as "the other side failed" would be generous to whoever cited.
            if _supported(t) and a["support_score"] < 1:
                attn_better.append(row)
            elif _supported(a) and t["support_score"] < 1:
                asked_better.append(row)

    # Featured: the clearest cases to put in front of a reader — the losing side
    # produced a citation (so the contrast is "cited the wrong thing", not "cited
    # nothing"), attention cited exactly one sentence, and the statement is long
    # enough to be meaningful but short enough to quote.
    def featured_key(r: dict):
        return (
            not r["asked_citation"],
            len(r["tokenpath_citation"]) != 1,
            abs(len(r["statement"]) - 130),
        )

    # Spread the featured picks across datasets. gov_report is 85% of the
    # candidates (its summaries carry far more statements), so an unconstrained
    # pick is all gov_report — representative of the count, useless as illustration.
    featured: list[dict] = []
    for ds in sorted({r["dataset"] for r in attn_better}):
        pool = sorted([r for r in attn_better if r["dataset"] == ds], key=featured_key)
        if pool:
            featured.append(pool[0])
    featured = sorted(featured, key=featured_key)[:n_featured]

    return {
        "featured": featured,
        "count_total": len(attn_better),
        "all_candidates": attn_better,
        # Reported even though it cuts against us.
        "count_asked_better": len(asked_better),
        "asked_better_candidates": asked_better,
        "provenance": {
            "split": split,
            "seed": seed,
            "asked_condition": "prompted_thinking",
            "statement_pairs_compared": n_pairs,
            "statement_pairs_dropped_misaligned": n_misaligned,
            "statement_pairs_dropped_as_fragments": n_fragments,
            "supported_means": "judge returned [[Fully supported]] (1.0) AND the "
                               "statement carried at least one citation",
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Exp 4 qualitative examples for the blog")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--featured", type=int, default=3)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    data = build(args.split, args.seed, args.featured)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    p = data["provenance"]
    print(f"statement pairs compared: {p['statement_pairs_compared']} "
          f"(dropped misaligned: {p['statement_pairs_dropped_misaligned']})")
    print(f"attention supported, asked not: {data['count_total']}")
    print(f"asked supported, attention not: {data['count_asked_better']}")
    print(f"featured: {len(data['featured'])}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

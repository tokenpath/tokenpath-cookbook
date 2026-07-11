"""Pluggable attention->citation aggregation.

Given a heatmap and one answer statement, turn attention mass into a set of
cited document sentences. Every strategy we hill-climb is a `cfg` dict here, so
the offline sweep and the production TokenPath method share one code path.

The baseline (cfg=BASELINE) reproduces the original `Heatmap.mass_to_sentences`:
sum the statement's answer-token rows, pool onto document sentences, keep the
top sentences whose share of the statement's mass clears a threshold.

Knobs (all optional; defaults = baseline):
  pool        "sum" | "max" | "mean"   how to combine answer tokens' votes per sentence
  row_norm    bool                     L1-normalize each answer token's row first
                                        (one token, one vote — stops a few high-magnitude
                                        tokens from dominating)
  atw         "uniform" | "content"    answer-token weights ("content" zeros stopwords/punct)
  sent_norm   "none" | "len" | "sqrt"  divide a sentence's score by its length (fights the
                                        long-sentence bias -> precision)
  min_conc    float                    drop sentences whose mass is too diffuse (per-token
                                        concentration below this) -> precision
  threshold   float                    keep sentences with mass-share >= this
  max_spans   int                      cap kept sentences (by score)
  merge_adjacent bool                  merge kept sentences <= merge_gap apart into a passage
  merge_gap   int                      sentence-index gap allowed when merging
  top1_fallback bool                   if nothing clears threshold, keep the single top sentence
"""

from __future__ import annotations

import numpy as np

BASELINE = {
    "pool": "sum", "row_norm": False, "atw": "uniform", "sent_norm": "none",
    "min_conc": 0.0, "threshold": 0.15, "rel_threshold": 0.0, "max_spans": 4,
    "merge_adjacent": False, "merge_gap": 1, "top1_fallback": False,
}

_STOP = set(
    "the a an of to in and or is are was were be been being am to that this these those with "
    "for on at by from as it its their his her they he she we you i which who whom whose when "
    "where how what why into over under between per not no also but if then than can may will "
    "would should could has have had do does did".split()
)


def _answer_weights(hm, rows, answer_text, mode):
    w = np.ones(len(rows), dtype=float)
    if mode == "content":
        for i, r in enumerate(rows):
            cs, ce = hm.answer_offsets[r]
            tok = answer_text[cs:ce].strip().lower().strip(".,;:()[]\"'`-—")
            if (not tok) or tok in _STOP or not any(c.isalnum() for c in tok):
                w[i] = 0.0
        if w.sum() == 0:  # degenerate (all stopwords) -> fall back to uniform
            w[:] = 1.0
    return w


def aggregate(hm, start, end, doc_sentence_spans, cfg=None, answer_text=None):
    """Return [(char_start, char_end, mass_share)] cited sentences for a statement.

    `answer_text` is required for content-word weighting; if None, uniform.
    Scores are normalized to shares of the statement's total, so `threshold` keeps
    the same 'fraction of mass' meaning across strategies.
    """
    c = {**BASELINE, **(cfg or {})}
    n_sent = len(doc_sentence_spans)
    rows = hm.answer_token_rows(start, end)
    if not rows or n_sent == 0:
        return []
    tok_sent = hm._token_sentence_index(doc_sentence_spans)
    valid = tok_sent >= 0
    ts = tok_sent[valid]

    # per-(answer-token, sentence) mass matrix M[len(rows), n_sent]
    M = np.zeros((len(rows), n_sent), dtype=float)
    for i, r in enumerate(rows):
        row = hm.matrix[r]
        if c["row_norm"]:
            s = row.sum()
            if s > 0:
                row = row / s
        np.add.at(M[i], ts, row[valid])

    w = _answer_weights(hm, rows, answer_text, c["atw"]) if answer_text is not None else np.ones(len(rows))
    M = M * w[:, None]

    if c["pool"] == "max":
        score = M.max(axis=0)
    elif c["pool"] == "mean":
        denom = max(float((w > 0).sum()), 1.0)
        score = M.sum(axis=0) / denom
    else:  # sum
        score = M.sum(axis=0)

    # sentence-length normalization (doc tokens per sentence)
    if c["sent_norm"] in ("len", "sqrt"):
        sent_len = np.zeros(n_sent)
        np.add.at(sent_len, ts, 1.0)
        sent_len = np.maximum(sent_len, 1.0)
        score = score / (sent_len if c["sent_norm"] == "len" else np.sqrt(sent_len))

    total = score.sum()
    if total <= 0:
        return []
    share = score / total  # fraction of the statement's (post-transform) mass per sentence

    # Concentration heuristic: omit sentences whose mass is spread thin across
    # their own document tokens; evaluate its effect with citation precision.
    if c["min_conc"] > 0:
        pooled = np.zeros(hm.matrix.shape[1])
        for r in rows:
            pooled += hm.matrix[r]
        keep_conc = np.ones(n_sent, dtype=bool)
        for j in range(n_sent):
            if share[j] <= 0:
                continue
            dt = np.where(ts == j)[0]  # positions into `valid`/doc-token space
            doc_toks = np.where(valid)[0][dt]
            if len(doc_toks) <= 1:
                continue
            m = pooled[doc_toks]
            if m.sum() <= 0 or float(m.max() / m.sum()) < c["min_conc"]:
                keep_conc[j] = False
        share = np.where(keep_conc, share, 0.0)

    # selection: absolute mass-share threshold, and/or a relative one (keep
    # sentences within `rel_threshold` of the top sentence — adapts to how many
    # sentences actually support the statement instead of a fixed cutoff).
    keep_mask = share >= c["threshold"]
    if c.get("rel_threshold", 0.0) > 0 and share.max() > 0:
        keep_mask &= share >= c["rel_threshold"] * share.max()
    kept = np.where(keep_mask)[0]
    if len(kept) == 0:
        if c["top1_fallback"]:
            kept = np.array([int(share.argmax())])
        else:
            return []
    order = kept[np.argsort(share[kept])[::-1]]

    if c["merge_adjacent"]:
        order_sorted = sorted(int(j) for j in order)
        runs, cur = [], [order_sorted[0]]
        for j in order_sorted[1:]:
            if j - cur[-1] <= c["merge_gap"]:
                cur.append(j)
            else:
                runs.append(cur); cur = [j]
        runs.append(cur)
        spans = []
        for run in runs:
            cs = doc_sentence_spans[run[0]][0]
            ce = doc_sentence_spans[run[-1]][1]
            spans.append((int(cs), int(ce), float(share[run].sum())))
        spans.sort(key=lambda x: x[2], reverse=True)
        return spans[: c["max_spans"]]

    order = order[: c["max_spans"]]
    return [(int(doc_sentence_spans[j][0]), int(doc_sentence_spans[j][1]), float(share[j]))
            for j in order]

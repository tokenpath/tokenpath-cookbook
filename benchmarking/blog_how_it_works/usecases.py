"""Three things attention-based attribution does that string matching can't.

Every example is a real call to the hosted TokenPath API. These are the cases
where lexical / retrieval citation breaks down: the answer and its source share
no tokens (cross-lingual, paraphrase), or the matching token appears many times
(dense tables). Run it and read what it points to.

    pip install requests numpy
    export TOKENPATH_API_KEY=...        # free key: https://platform.tokenpath.ai
    python usecases.py

Two endpoints are used:
  /v1/attributions          one answer span -> its single best source span
  /v1/attributions/heatmap  the full answer-token x doc-token matrix (we roll it
                            up to whole sentences, robust for paraphrase)
"""

from __future__ import annotations

import os
import re

import numpy as np
import requests

BASE = "https://api.tokenpath.ai"
KEY = os.environ["TOKENPATH_API_KEY"]
AUTH = {"Authorization": f"Bearer {KEY}"}


def attribute_span(document, question, answer, phrase):
    """Resolve one answer phrase to its single best source span (word-level)."""
    start = answer.index(phrase)
    r = requests.post(f"{BASE}/v1/attributions", headers=AUTH, timeout=60,
                      json={"document": document, "question": question,
                            "answer": answer, "spans": [[start, start + len(phrase)]]})
    r.raise_for_status()
    return r.json()["spans"][0]["source"]


def cite_sentences(document, question, answer, top_k=2):
    """Roll the whole answer's attention up to document sentences (heatmap)."""
    r = requests.post(f"{BASE}/v1/attributions/heatmap", headers=AUTH, timeout=60,
                      json={"document": document, "question": question, "answer": answer})
    r.raise_for_status()
    h = r.json()
    M = np.zeros(h["shape"])
    M[h["row"], h["col"]] = h["data"]
    mass = M.sum(0)
    mass = mass / (mass.sum() or 1.0)                      # distribution over doc tokens
    sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]+", document)]
    per_sent = np.zeros(len(sents))
    for (cs, ce), m in zip(h["document_offsets"], mass):
        mid = (cs + ce) / 2
        for si, (ss, se) in enumerate(sents):
            if ss <= mid < se:
                per_sent[si] += m
                break
    order = np.argsort(per_sent)[::-1][:top_k]
    return [(document[sents[i][0]:sents[i][1]].strip(), float(per_sent[i])) for i in order]


if __name__ == "__main__":
    # 1. MULTILINGUAL — English answer grounded in a German source. No shared tokens.
    print("1. Multilingual — English answer, German source")
    doc = ("Die Zugspitze ist mit 2.962 Metern der höchste Berg Deutschlands. "
           "Sie liegt in den Bayerischen Alpen an der Grenze zu Österreich.")
    ans = "The Zugspitze is 2,962 meters tall."
    src = attribute_span(doc, "How tall is the Zugspitze?", ans, "2,962 meters")
    print(f"   answer span '2,962 meters'")
    print(f"   ↳ source {src['text']!r}  (conf {src['confidence']:.2f})  — the German number\n")

    # 2. PARAPHRASE — "Senate confirmation" appears nowhere in the source.
    print("2. Paraphrase — abstractive restatement (sentence-level)")
    doc = ("Under the Constitution, principal officers are appointed by the President "
           "by and with the advice and consent of the Senate. "
           "Inferior officers may be appointed by department heads alone.")
    ans = "Principal officers require Senate confirmation."
    print(f"   answer {ans!r}")
    for sent, share in cite_sentences(doc, "How are principal officers appointed?", ans):
        print(f"   [{share:5.1%}] {sent}")
    print("   — mass concentrates on the 'advice and consent' sentence, not the distractor\n")

    # 3. TABLES — pin the exact cell among many similar-looking numbers.
    print("3. Tables — the exact cell, not a similar number")
    doc = ("Quarterly revenue by segment ($M):\n"
           "Cloud:    Q3 47.1   Q4 52.6\n"
           "Hardware: Q3 31.8   Q4 29.4\n"
           "Services: Q3 18.2   Q4 20.5\n")
    ans = "Cloud's Q4 revenue was $52.6M."
    src = attribute_span(doc, "What was Cloud Q4 revenue?", ans, "$52.6M")
    ctx = doc[max(0, src["start"] - 16):src["end"]].replace("\n", " ⏎ ")
    print(f"   answer span '$52.6M'")
    print(f"   ↳ source {src['text']!r}  (conf {src['confidence']:.2f})  — in row: …{ctx}…")

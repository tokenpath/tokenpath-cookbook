# How TokenPath works — a hands-on cookbook

*TokenPath maps any span of an LLM's answer back to the exact source spans that
produced it — measured from an open model's attention, after generation, on any
model's output. This post is mostly code: we'll call the API, watch attention
turn into citations, do three things string matching can't, and then drop down to
raw Transformers to prove the signal is really the model's own attention.*

> **Updated 2026-07-15.** The hosted API's open reference model was upgraded
> (the API sections' outputs below were re-run against the current API on that
> date — confidences and heatmap shapes shifted a little, every conclusion
> holds). The raw-Transformers section is pinned to Llama-3.1-8B-Instruct and
> is unaffected.

> **Run it yourself.** Everything here is in a runnable notebook —
> [`how-tokenpath-works.ipynb`](../../notebooks/how-tokenpath-works.ipynb)
> ([open in Colab](https://colab.research.google.com/github/tokenpath/tokenpath-cookbook/blob/main/notebooks/how-tokenpath-works.ipynb)).
> The API sections need only a free `TOKENPATH_API_KEY`; the local-model section
> is optional and guarded.

---

## The one idea

When a model writes an answer from a document, it doesn't read the whole document
uniformly — at each generated token, attention concentrates on the few source
tokens it's actually drawing from. That concentration isn't a metaphor; it's a
set of weights inside the forward pass. **TokenPath reads those weights and turns
them into citations**, post-hoc, on any model's output. Not "this paragraph looks
relevant" but *these six answer words came from those nine document words.*

Two consequences worth keeping in mind as you read the code:

- **It's a measurement, not a second opinion.** A prompted "add citations" pass
  asks an LLM to *guess* what supports what, after the fact. Attention is the
  mechanism the text was produced through.
- **It's model-agnostic.** Your answer can come from GPT-5.5, Claude, your
  fine-tune — TokenPath re-reads the `(document, answer)` pair through an open
  reference model and measures where attention lands. Which reference model is
  an API implementation detail that improves over time; §3 re-derives the
  mechanism on a pinned one (Llama-3.1-8B-Instruct) you can run yourself.

Let's watch it happen.

---

## 1. From attention to a citation

The `/v1/attributions/heatmap` endpoint returns the raw attention: a sparse
answer-token × document-token matrix.

```python
import os, numpy as np, requests
BASE = "https://api.tokenpath.ai"
AUTH = {"Authorization": f"Bearer {os.environ['TOKENPATH_API_KEY']}"}

def heatmap(document, question, answer):
    r = requests.post(f"{BASE}/v1/attributions/heatmap", headers=AUTH,
                      json={"document": document, "question": question, "answer": answer})
    r.raise_for_status()
    return r.json()

document = ("The Oregon Duck is the mascot of the University of Oregon. "
            "It is based on Disney's Donald Duck. "
            "The mascot wears a green and yellow costume and a green beanie cap. "
            "The costume was redesigned in 2002.")
h = heatmap(document, "What colors does the Oregon Duck wear?",
            "The Oregon Duck wears green and yellow.")
print("shape [answer_tokens, document_tokens]:", h["shape"])
print("non-zero attention entries:", len(h["data"]))
```
```
shape [answer_tokens, document_tokens]: [8, 46]
non-zero attention entries: 174
```

Densify it and pool each answer token's attention onto **document sentences** —
that's all "make a citation" is:

```python
import re

def cite_sentences(h, document, top_k=2):
    M = np.zeros(h["shape"]); M[h["row"], h["col"]] = h["data"]
    mass = M.sum(0); mass = mass / mass.sum()                 # distribution over doc tokens
    sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]+", document)]
    per = np.zeros(len(sents))
    for (cs, ce), m in zip(h["document_offsets"], mass):
        mid = (cs + ce) / 2
        for i, (s, e) in enumerate(sents):
            if s <= mid < e: per[i] += m; break
    order = np.argsort(per)[::-1][:top_k]
    return [(document[sents[i][0]:sents[i][1]].strip(), float(per[i])) for i in order]

for sent, share in cite_sentences(h, document):
    print(f"[{share:5.1%}] {sent}")
```
```
[75.5%] The mascot wears a green and yellow costume and a green beanie cap.
[17.4%] The Oregon Duck is the mascot of the University of Oregon.
```

Look at *which* sentence wins. The answer is about "the Oregon Duck," but the
color fact lives in a sentence that says **"The mascot."** Attention followed the
model's own coreference to the fact-bearing sentence — the exact case a keyword or
embedding search gets wrong (it would match the *name* sentence). This coreference
pattern is where TokenPath most cleanly beats retrieval, and it's all over the
[quality benchmark](../blog/post.md).

If you want a single best source span for a specific phrase instead of sentences,
that's the `/v1/attributions` endpoint:

```python
def attribute_span(document, question, answer, phrase):
    start = answer.index(phrase)
    r = requests.post(f"{BASE}/v1/attributions", headers=AUTH,
                      json={"document": document, "question": question,
                            "answer": answer, "spans": [[start, start + len(phrase)]]})
    r.raise_for_status()
    return r.json()["spans"][0]["source"]

src = attribute_span(document, "What colors does the Oregon Duck wear?",
                     "The Oregon Duck wears green and yellow.", "green and yellow")
print(src["text"], f"(confidence {src['confidence']:.2f})")
```
```
green and yellow (confidence 0.88)
```

---

## 2. Things string matching can't do

This is where attention earns its keep. Each example below is a real API call.

### 2.1 Disambiguation — the same date twice, one for each claim

An offer letter where **"March 2, 2026" appears twice** — once as the agreement's
effective date, once as the first day of work — and the answer restates both.
Keyword search sees two identical strings and can't tell which claim came from
which; attention sends each claim to the date it actually read.

```python
doc = ('This letter agreement (the "Agreement") is made effective as of March 2, 2026 '
       'between Acme Robotics, Inc. and Jordan Lee. You are offered the role of Senior '
       'Mechanical Engineer. Your first day of employment will be March 2, 2026.')
q = "When does the agreement start, and when is my first day of work?"
a = ("The agreement is effective as of March 2, 2026, and your first day of "
     "employment is March 2, 2026.")

# the answer states the same date twice — attribute each claim separately
d1 = a.index("March 2, 2026")             # the "effective as of" claim
d2 = a.index("March 2, 2026", d1 + 1)     # the "first day of employment" claim

def source_of(start, end):
    r = requests.post(f"{BASE}/v1/attributions", headers=AUTH,
                      json={"document": doc, "question": q, "answer": a,
                            "spans": [[start, end]]})
    return r.json()["spans"][0]["source"]

for label, s in [("effective-date claim", d1), ("first-day claim", d2)]:
    src = source_of(s, s + 13)
    print(f"{label:22} -> doc[{src['start']}:{src['end']}]  ...{doc[src['start']-27:src['end']]}...  "
          f"(conf {src['confidence']:.2f})")
```
```
effective-date claim   -> doc[64:77]  ...") is made effective as of March 2, 2026...  (conf 0.81)
first-day claim        -> doc[215:228]  ... day of employment will be March 2, 2026...  (conf 0.67)
```

Two identical answer dates, two **different** source spans — each claim resolved to
the occurrence it was actually generated from. A string/embedding search would map
both to the same match; attention disambiguates by the surrounding *meaning* the
model used. *(This works because each claim carries its own distinct context in the
answer. The harder sibling case — the same value in parallel rows with no
distinguishing context — is a known limitation; see the honest note under Tables.)*

### 2.2 Multilingual — the answer isn't in the source's language

```python
doc = ("Die Zugspitze ist mit 2.962 Metern der höchste Berg Deutschlands. "
       "Sie liegt in den Bayerischen Alpen an der Grenze zu Österreich.")
src = attribute_span(doc, "How tall is the Zugspitze?",
                     "The Zugspitze is 2,962 meters tall.", "2,962 meters")
print(src["text"], f"(confidence {src['confidence']:.2f})")
```
```
2.962 Metern (confidence 0.94)
```

An English answer, a German source, **zero shared tokens** — and it lands on
`2.962` (note the European decimal comma). The model represents the quantity, not
the string, so attribution crosses the language boundary. A keyword citer can't do
this at all.

### 2.3 Paraphrase — no lexical overlap to match on

The phrase *"Senate confirmation"* appears **nowhere** in the source, which says
*"advice and consent of the Senate."*

```python
doc = ("Under the Constitution, principal officers are appointed by the President "
       "by and with the advice and consent of the Senate. "
       "Inferior officers may be appointed by department heads alone.")
h = heatmap(doc, "How are principal officers appointed?",
            "Principal officers require Senate confirmation.")
for sent, share in cite_sentences(h, doc):
    print(f"[{share:5.1%}] {sent}")
```
```
[90.0%] Under the Constitution, principal officers are appointed by the President by and with the advice and consent of the Senate.
[10.0%] Inferior officers may be appointed by department heads alone.
```

**90.0%** of the answer's attention lands on the advice-and-consent sentence,
10.0% on the plausible distractor — despite the answer and source sharing only the
word "Senate." Attention connects them because the model *paraphrased from* that
span.

### 2.4 Tables — the exact cell, not a similar-looking number

```python
doc = ("Quarterly revenue by segment ($M):\n"
       "Cloud:    Q3 47.1   Q4 52.6\n"
       "Hardware: Q3 31.8   Q4 29.4\n"
       "Services: Q3 18.2   Q4 20.5\n")
src = attribute_span(doc, "What was Cloud Q4 revenue?",
                     "Cloud's Q4 revenue was $52.6M.", "$52.6M")
print(src["text"], f"(confidence {src['confidence']:.2f})")
```
```
52.6 (confidence 0.85)
```

A grid full of similar-looking numbers, and it pins the **exact cell** — the
Cloud/Q4 intersection — not the other `52.6`-ish values. A chunk retriever would
cite the whole table; string search can't tell one number from another.

> **Honest note on repeated values.** If a value is *byte-for-byte identical*
> across rows (two segments both exactly `$4.2M`), attribution tends to resolve to
> the first occurrence rather than reliably picking the queried row — the signal is
> genuinely ambiguous when the tokens are. It disambiguates by *content*, so it
> shines when the cells differ, not when they're duplicates.

---

## 3. Is this really the model's attention?

Everything above went through the hosted API. Fair to be skeptical — so here's the
same measurement from raw 🤗 Transformers, ~40 lines, no TokenPath involved. This
section is pinned to **Llama-3.1-8B-Instruct** — the reference model the API served
at launch — so it reproduces exactly even as the hosted API's reference model
changes. This is **not** how the hosted API is implemented (that has to
be fast on 100k-token documents — see §3.3); it's the simplest thing that computes
the same number, so you can check it.

### 3.1 Get the attention map yourself

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"     # any 8B-Instruct checkpoint
tok = AutoTokenizer.from_pretrained(LLAMA)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, dtype=torch.bfloat16, attn_implementation="eager").eval()

# one forward pass over [framing][document][framing][answer], attention exposed
out = model(input_ids, output_attentions=True, use_cache=False)
# out.attentions: tuple(num_layers) of [batch, num_heads, seq, seq]

attn = torch.stack([out.attentions[l][0, h]                    # average a band of
                    for l in range(14, 24)                     # mid/late layers,
                    for h in range(out.attentions[l].shape[1])]).float().mean(0)
sub = attn[answer_positions][:, document_positions]            # [n_ans, n_doc]
sub = sub / sub.sum(-1, keepdim=True)                          # a distribution per answer token
```

Roll that up to sentences exactly as before, and on the Oregon Duck example you
get:

```
attention matrix: (8, 45) (answer tokens × doc tokens)
Top cited sentences (answer-attention mass):
  [67.1%]  The mascot wears a green and yellow costume and a green beanie cap.
  [18.3%]  The Oregon Duck is the mascot of the University of Oregon.
```

Same verdict as the API: attention, read straight off Llama's forward pass, lands
on "The mascot" sentence. (The exact shares differ from §1 — that was the API's
current reference model, this is pinned Llama — but both put ~4× more mass on the
fact sentence than on the name sentence.) (Full runnable version in the notebook's optional
local-model section, and in [`reproduce_attention.py`](reproduce_attention.py).)

### 3.2 Which heads? Not all of them

If you average *all* heads the signal is mushy — most of the ~1024 heads in an 8B
model do local/syntactic work, not source attribution. Only a subset (cousins of
the *induction* / *retrieval* heads from the interpretability literature) actually
track "which source token is this answer drawing from." So we **probe**: bury a
known **needle** fact among distractors, ask about it, and measure per `(layer,
head)` how much of the answer's attention lands on the needle. Averaged over 24
synthetic examples on Llama-3.1-8B-Instruct:

```
Top attribution heads (mean needle-mass over 24 synthetic examples):
  layer 14  head 31   needle-mass 0.935
  layer 14  head 28   needle-mass 0.932
  layer 24  head 27   needle-mass 0.930
  layer 17  head 29   needle-mass 0.926
  ...
mean over ALL heads: 0.489      (5-sentence chance baseline ≈ 0.20)
```

The best heads put **~0.93** of the answer's attention on the fact sentence vs
~0.49 for the average head — and they cluster in the middle-to-late layers, which
is exactly the band we averaged over in §3.1. (Runnable, on synthetic data you can
read, in [`find_attribution_heads.py`](find_attribution_heads.py). Production head
selection is this same idea run as a much larger, careful battery, combined with a
learned weighting rather than a flat average — but the shape is exactly this: find
the heads that attribute, ignore the rest. We don't disclose the shipped set. It's
also what makes the reference model swappable: point the battery at a new open
model and it finds that model's attribution heads — which is how the API's
2026-07 reference-model upgrade was qualified.)

### 3.3 The catch: long context

You'll have noticed `attn_implementation="eager"`. That's the part that makes the
naive version useless in production. **FlashAttention** (and fused SDPA) compute
`softmax(QKᵀ)V` *without ever materializing* the `seq × seq` score matrix — that's
what makes 100k-token context affordable. But it means there's no attention matrix
to read: `output_attentions=True` silently falls back to the eager path that
builds the full matrix, which for a long document is billions of floats per layer.
You cannot just flip the flag at scale.

The way out is that we need almost none of that matrix: only **answer-token rows**
(answers are short), for a **handful of selected heads** (§3.2), against
**document-token columns**. That's a thin slice. You can compute exactly it with
the same tiled online-softmax math FlashAttention uses — stream over document-key
tiles, but *keep* the answer-query score rows instead of discarding them after the
`V` multiply — so memory stays proportional to `answer_tokens × tile`, not `seq²`.
The generation pass is untouched; attribution is a cheap second read. That's the
engineering the hosted API invests in; the toy script above skips it and pays
O(n²). (Kernel background: the [FlashAttention paper](https://arxiv.org/abs/2205.14135).)

### 3.4 From heatmap to clean citations

The last mile is aggregation: pool each statement's mass onto sentences,
one-token-one-vote normalize (so a couple of high-magnitude tokens can't
dominate), threshold, and merge adjacent supporting sentences into passages. We
tuned that carefully — it's worth ~6 F1 points on the benchmark — and the exact
recipe plus the offline sweep is open in [`benchmarking/`](../). The attribution
itself doesn't change; aggregation only decides how to *report* it.

---

## 4. How good is it, and what does it measure?

**Quality.** On [LongBench-Cite](https://github.com/THUDM/LongCite) (a benchmark
we didn't write), post-hoc attention attribution now matches generation-time
citation quality — **F1 0.815**, ahead of Anthropic's Citations API (0.812), with
only a prompted frontier LLM ahead (0.851) — while being **~6× faster, ~7×
cheaper, needing no document index, and working on any model's output.** It beats
naive retrieval (0.62) outright, with the highest precision of any method (0.94). Full methodology, per-dataset numbers, the
cost/quality frontier, and honest limitations:
**[How good are post-hoc citations? →](../blog/post.md)**

**What it measures.** TokenPath re-reads the `(document, question, answer)` tuple
with an open reference model and uses attention to associate answer spans with
source spans. The output is citation-oriented source attribution: cited text,
character offsets, and an attribution score that applications can render or
aggregate. LongBench-Cite evaluates those source associations as citations using
recall, precision, and F1.

The important scope is that TokenPath reads a *reference* open model's attention,
not necessarily the exact model that wrote the answer. The result is therefore a
post-hoc source association rather than a literal trace of the original generation.
That same setup is what lets it attribute output from any model without changing
the generation stack.

---

## Run it yourself

- **Notebook:** [`how-tokenpath-works.ipynb`](../../notebooks/how-tokenpath-works.ipynb)
  — everything above, runnable. API sections need only a free key; the local-model
  section is optional and guarded.
- **Minimal scripts:** [`reproduce_attention.py`](reproduce_attention.py) (attention → citation on Llama)
  and [`find_attribution_heads.py`](find_attribution_heads.py) (the head probe).
- **Benchmark:** [`benchmarking/`](../) and the [quality post](../blog/post.md).
- **Free key:** [platform.tokenpath.ai](https://platform.tokenpath.ai) — 10M
  attributed tokens, then $1 / 1M.

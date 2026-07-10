"""Minimal, from-scratch reproduction of what TokenPath measures.

This is deliberately the *simplest* thing that works — plain 🤗 Transformers with
`output_attentions=True`, no kernels, no batching, no cleverness. It is not how
the hosted API is implemented (that has to be fast on 100k-token documents; see
the "long sequences" section of the blog post). But it computes the *same
quantity*: for each answer token, how much the generating model attended to each
document token, read off Llama-3.1-8B-Instruct's own attention. Run it and you
can watch attention land on the sentence that actually supports a claim.

    pip install "transformers>=4.44" torch accelerate
    python reproduce_attention.py

The model is gated on the Hub — `huggingface-cli login` once, with access to
meta-llama/Llama-3.1-8B-Instruct. Runs on a single 16GB+ GPU (bf16); CPU works
for the tiny document below but is slow.
"""

from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Any Llama-3.1-8B-Instruct checkpoint. Override with LLAMA_MODEL to point at a
# local path or an ungated mirror if you don't have meta-llama gated access.
MODEL = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Llama-3.1's chat framing, written out as literal text so we control exactly
# where the document and the answer sit in the token stream and can recover
# character offsets for each. (apply_chat_template would produce the same tokens;
# we spell it out only to keep the offset bookkeeping transparent.)
PROMPT_PRE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "Answer the question using only the document.\n\nQuestion: {question}\n\nDocument:\n"
)
PROMPT_MID = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
PROMPT_END = "<|eot_id|>"

# Attribution is strongest in the middle-to-late layers, and only in a subset of
# heads (see find_attribution_heads.py for how to discover them on synthetic
# data). Here we simply average a band of upper-middle layers over all heads —
# crude, but enough to see the effect. Set HEADS to a list of (layer, head) to
# use a hand-picked head subset instead.
LAYER_BAND = range(14, 24)   # 0-indexed; Llama-3.1-8B has 32 layers
HEADS: list[tuple[int, int]] | None = None


def encode_segment(tok, text):
    """Token ids + per-token (char_start, char_end) offsets into `text`."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def build_inputs(tok, document, question, answer):
    """Concatenate the framed prompt and record where doc/answer tokens land.

    Returns input_ids and, for the document and the answer separately, the list
    of (sequence_position, char_start, char_end) so we can map any token back to
    its characters in the original string.
    """
    ids, doc_map, ans_map = [], [], []

    ids += encode_segment(tok, PROMPT_PRE.format(question=question))[0]

    doc_ids, doc_off = encode_segment(tok, document)
    for pos, (cs, ce) in zip(range(len(ids), len(ids) + len(doc_ids)), doc_off):
        doc_map.append((pos, cs, ce))
    ids += doc_ids

    ids += encode_segment(tok, PROMPT_MID)[0]

    ans_ids, ans_off = encode_segment(tok, answer)
    for pos, (cs, ce) in zip(range(len(ids), len(ids) + len(ans_ids)), ans_off):
        ans_map.append((pos, cs, ce))
    ids += ans_ids

    ids += encode_segment(tok, PROMPT_END)[0]
    return ids, doc_map, ans_map


@torch.no_grad()
def attention_matrix(model, tok, document, question, answer):
    """[n_answer_tokens, n_doc_tokens] attention, plus the two offset maps.

    matrix[i, j] = how much answer token i attended to document token j,
    averaged over the selected layers/heads and renormalized over doc tokens.
    """
    ids, doc_map, ans_map = build_inputs(tok, document, question, answer)
    input_ids = torch.tensor([ids], device=model.device)

    out = model(input_ids, output_attentions=True, use_cache=False)
    # out.attentions: tuple(num_layers) of [batch, num_heads, seq, seq].
    if HEADS is not None:
        chosen = torch.stack([out.attentions[l][0, h] for l, h in HEADS])
    else:
        chosen = torch.stack([out.attentions[l][0, h]
                              for l in LAYER_BAND
                              for h in range(out.attentions[l].shape[1])])
    attn = chosen.float().mean(0)  # [seq, seq], averaged over chosen heads

    ans_pos = [p for p, _, _ in ans_map]
    doc_pos = [p for p, _, _ in doc_map]
    sub = attn[ans_pos][:, doc_pos]              # [n_ans, n_doc]
    sub = sub / sub.sum(-1, keepdim=True).clamp_min(1e-9)   # renormalize over doc
    return sub.cpu(), doc_map, ans_map


def cite_sentences(matrix, doc_map, document, top_k=2):
    """Roll answer→doc-token attention up to whole document sentences.

    Sum the whole answer's mass onto each sentence (by which sentence each doc
    token's midpoint falls in) and return the highest-mass sentences. This is the
    same pooling the benchmark harness does — see common/aggregate.py.
    """
    sent_spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]+|\S[^.!?]*$", document)]
    mass = matrix.sum(0)             # over all answer tokens -> [n_doc]
    mass = mass / mass.sum()         # renormalize to a distribution (rows summed to n_ans)
    per_sent = [0.0] * len(sent_spans)
    for (_, cs, ce), m in zip(doc_map, mass.tolist()):
        mid = (cs + ce) / 2
        for si, (ss, se) in enumerate(sent_spans):
            if ss <= mid < se:
                per_sent[si] += m
                break
    ranked = sorted(range(len(sent_spans)), key=lambda i: per_sent[i], reverse=True)
    return [(document[sent_spans[i][0]:sent_spans[i][1]].strip(), per_sent[i])
            for i in ranked[:top_k]]


if __name__ == "__main__":
    # A coreference case — the color fact lives in a sentence that says "The
    # mascot", not "the Oregon Duck". Lexical retrieval matches the *name*
    # sentence; attention follows the model's own coreference to the *fact*.
    document = (
        "The Oregon Duck is the mascot of the University of Oregon. "
        "It is based on Disney's Donald Duck. "
        "The mascot wears a green and yellow costume and a green beanie cap. "
        "The costume was redesigned in 2002."
    )
    question = "What colors does the Oregon Duck wear?"
    answer = "The Oregon Duck wears green and yellow."

    print(f"loading {MODEL} …")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",   # REQUIRED: flash/sdpa never build the matrix
    )
    model.eval()

    matrix, doc_map, _ = attention_matrix(model, tok, document, question, answer)
    print(f"\nattention matrix: {tuple(matrix.shape)} (answer tokens × doc tokens)\n")
    print("Top cited sentences (answer-attention mass):")
    for sent, share in cite_sentences(matrix, doc_map, document):
        print(f"  [{share:5.1%}]  {sent}")

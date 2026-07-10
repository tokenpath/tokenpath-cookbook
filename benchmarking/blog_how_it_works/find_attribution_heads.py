"""Discover which attention heads *attribute*, on synthetic data.

Not every head is useful for attribution. Most do local/syntactic things; only
some track "which source token is this answer token actually drawing from."
Rather than guess, we probe: build documents with a known **needle** sentence
that contains the fact, surround it with distractors, ask about the fact, and
measure — per (layer, head) — how much of the answer's attention lands on the
needle. Heads with high needle-mass across many random examples are attribution
heads.

This script uses *synthetic, self-generated* data (no proprietary corpus and no
leak of which heads the hosted API actually ships — production selection uses a
much larger, more careful battery). It's the methodology, reproducibly, on data
you can read.

    pip install "transformers>=4.44" torch accelerate
    python find_attribution_heads.py
"""

from __future__ import annotations

import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduce_attention import PROMPT_END, PROMPT_MID, PROMPT_PRE, encode_segment

MODEL = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
N_EXAMPLES = 24

# Synthetic fact templates: (subject, relation phrasing, value, question, answer).
FACTS = [
    ("The Zorvan reactor", "operates at a core temperature of", "812 kelvin",
     "What temperature does the Zorvan reactor operate at?",
     "The Zorvan reactor operates at 812 kelvin."),
    ("The city of Brelmont", "was founded in the year", "1643",
     "When was Brelmont founded?", "Brelmont was founded in 1643."),
    ("The Kessler comet", "completes one orbit every", "47 years",
     "How often does the Kessler comet orbit?",
     "The Kessler comet orbits every 47 years."),
    ("The Vantor protocol", "encrypts data using a", "512-bit key",
     "What key size does the Vantor protocol use?",
     "The Vantor protocol uses a 512-bit key."),
    ("The Marlow bridge", "spans a distance of", "2.3 kilometers",
     "How long is the Marlow bridge?", "The Marlow bridge is 2.3 kilometers long."),
]
DISTRACTORS = [
    "The weather that season was unusually mild across the region.",
    "Several committees reviewed the proposal before it was published.",
    "Local historians disagree about many unrelated details.",
    "The archive contains thousands of documents from that period.",
    "Funding came from a mix of public and private sources.",
    "Visitors often remark on the surrounding scenery.",
]


def make_example(rng):
    subj, rel, val, question, answer = rng.choice(FACTS)
    needle = f"{subj} {rel} {val}."
    distractors = rng.sample(DISTRACTORS, k=4)
    sentences = distractors[:2] + [needle] + distractors[2:]
    document = " ".join(sentences)
    needle_start = document.index(needle)
    needle_span = (needle_start, needle_start + len(needle))
    return document, question, answer, needle_span


def build_inputs(tok, document, question, answer):
    ids, doc_map, ans_pos = [], [], []
    ids += encode_segment(tok, PROMPT_PRE.format(question=question))[0]
    doc_ids, doc_off = encode_segment(tok, document)
    for pos, (cs, ce) in zip(range(len(ids), len(ids) + len(doc_ids)), doc_off):
        doc_map.append((pos, cs, ce))
    ids += doc_ids
    ids += encode_segment(tok, PROMPT_MID)[0]
    ans_ids, _ = encode_segment(tok, answer)
    ans_pos = list(range(len(ids), len(ids) + len(ans_ids)))
    ids += ans_ids
    ids += encode_segment(tok, PROMPT_END)[0]
    return ids, doc_map, ans_pos


@torch.no_grad()
def needle_mass_per_head(model, tok, document, question, answer, needle_span):
    """For every (layer, head): fraction of answer→doc attention on the needle."""
    ids, doc_map, ans_pos = build_inputs(tok, document, question, answer)
    input_ids = torch.tensor([ids], device=model.device)
    out = model(input_ids, output_attentions=True, use_cache=False)

    ns, ne = needle_span
    doc_positions = [p for p, _, _ in doc_map]
    is_needle = torch.tensor([ns <= (cs + ce) / 2 < ne for _, cs, ce in doc_map])

    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    scores = torch.zeros(n_layers, n_heads)
    for l in range(n_layers):
        a = out.attentions[l][0].float()            # [heads, seq, seq]
        sub = a[:, ans_pos][:, :, doc_positions]     # [heads, n_ans, n_doc]
        sub = sub / sub.sum(-1, keepdim=True).clamp_min(1e-9)
        per_head = sub.mean(1)                        # avg over answer tokens
        scores[l] = per_head[:, is_needle].sum(-1)    # mass on needle
    return scores


if __name__ == "__main__":
    rng = random.Random(0)
    print(f"loading {MODEL} …")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    total = None
    for i in range(N_EXAMPLES):
        doc, q, a, needle = make_example(rng)
        s = needle_mass_per_head(model, tok, doc, q, a, needle)
        total = s if total is None else total + s
        print(f"  example {i + 1}/{N_EXAMPLES}", end="\r")
    mean = total / N_EXAMPLES
    print("\n\nTop attribution heads (mean needle-mass over "
          f"{N_EXAMPLES} synthetic examples):")
    flat = [(l, h, mean[l, h].item())
            for l in range(mean.shape[0]) for h in range(mean.shape[1])]
    for l, h, v in sorted(flat, key=lambda x: x[2], reverse=True)[:15]:
        print(f"  layer {l:2d}  head {h:2d}   needle-mass {v:.3f}")
    print(f"\nmean over ALL heads: {sum(v for _, _, v in flat) / len(flat):.3f}")
    print("Random-baseline needle-mass ≈ 1/num_sentences. The top heads sit far "
          "\nabove both — those are the ones worth reading attention from.")

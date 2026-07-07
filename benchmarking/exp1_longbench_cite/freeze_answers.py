"""Freeze ONE answer set — the fair-comparison cornerstone.

Every post-hoc method must attribute the *same* answers, or we would be
comparing generators, not attribution. So we generate one answer per example
with a single generator, WITHOUT citations, and freeze it to
results/exp1_frozen_answers.jsonl. TokenPath, the prompted baseline, and the
embedding baseline all consume these exact strings.

(The Anthropic Citations API baseline is the one exception: it regenerates the
answer with citations. That asymmetry — and its answer-preservation rate against
this frozen set — is a first-class column in the results table, not a footnote.)

Deterministic: temperature 0, pinned generator model, resumable via idx-dedup.
"""

from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from .. import config
from ..common import env
from ..common.io_utils import append_jsonl, load_done_ids
from . import load_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FROZEN_PATH = os.path.join(RESULTS_DIR, "exp1_frozen_answers.jsonl")

GEN_SYSTEM = (
    "You answer questions strictly from the provided document. Write a direct, "
    "self-contained answer in complete sentences. Do NOT add citations, "
    "footnotes, bracketed markers, or references to 'the document' — just state "
    "the facts. Do not include information that is not in the document."
)


def gen_prompt(context: str, query: str) -> list[dict]:
    user = (
        f"<document>\n{context}\n</document>\n\n"
        f"<question>\n{query}\n</question>\n\n"
        "Answer the question using only the document above."
    )
    return [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": user},
    ]


def freeze(examples: list[dict], model: str, out_path: str = FROZEN_PATH) -> None:
    client = env.openrouter_client()
    done = load_done_ids(out_path)
    todo = [e for e in examples if e["idx"] not in done]
    print(f"freeze_answers: {len(done)} cached, {len(todo)} to generate ({model})")
    for e in tqdm(todo):
        res = client.chat(model, gen_prompt(e["context"], e["query"]), temperature=0.0)
        append_jsonl(
            out_path,
            {
                "idx": e["idx"],
                "dataset": e["dataset"],
                "language": e.get("language"),
                "query": e["query"],
                "prediction": res.text.strip(),
                "gen_model": res.model,
                "gen_prompt_tokens": res.prompt_tokens,
                "gen_completion_tokens": res.completion_tokens,
                "gen_cost_usd": res.cost_usd,
                "gen_seconds": res.seconds,
            },
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Freeze one no-citation answer per example")
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-languages", action="store_true")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--model", default=config.GENERATOR_MODEL)
    args = ap.parse_args()

    names = ["val", "test"] if args.split == "both" else [args.split]
    examples: list[dict] = []
    for name in names:
        examples += load_data.load_split(
            name,
            seed=args.seed,
            english_only=not args.all_languages,
            limit_per_dataset=args.limit_per_dataset,
        )
    freeze(examples, model=args.model)

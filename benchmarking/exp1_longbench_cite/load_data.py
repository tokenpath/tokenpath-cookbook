"""Load LongBench-Cite and split it into val / test.

Data source: THUDM/LongCite, LongBench-Cite/LongBench-Cite.json (1000 examples
across longbench-chat, hotpotqa, gov_report, multifieldqa_en/zh, dureader).
Downloaded once to benchmarking/data/ and cached.

Discipline (stated in the post): TokenPath's mass threshold is tuned on the VAL
split only and reported on TEST. The split is a deterministic, per-dataset
shuffle keyed by seed so it reproduces exactly and never leaks test into tuning.

Language: the blog is English-facing, so we default to English datasets. The
`--languages` flag can re-include the Chinese ones (dureader, multifieldqa_zh)
to reproduce the full published average.
"""

from __future__ import annotations

import argparse
import os
import random
import urllib.request

from ..common.io_utils import ensure_dir, read_json, write_json

DATA_URL = (
    "https://raw.githubusercontent.com/THUDM/LongCite/main/"
    "LongBench-Cite/LongBench-Cite.json"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "LongBench-Cite.json")

ENGLISH_DATASETS = {"longbench-chat", "hotpotqa", "gov_report", "multifieldqa_en"}
# eval_cite.py excludes these two from its reported average:
EXCLUDED_FROM_AVG = {"multifieldqa_en", "multifieldqa_zh"}


def download(force: bool = False) -> str:
    ensure_dir(DATA_DIR)
    if force or not os.path.exists(RAW_PATH):
        print(f"Downloading LongBench-Cite -> {RAW_PATH}")
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    return RAW_PATH


def load_raw() -> list[dict]:
    if not os.path.exists(RAW_PATH):
        download()
    return read_json(RAW_PATH)


def split(
    examples: list[dict],
    val_frac: float = 0.2,
    seed: int = 0,
    languages: set[str] | None = None,
    datasets: set[str] | None = None,
    limit_per_dataset: int | None = None,
) -> dict[str, list[dict]]:
    """Return {'val': [...], 'test': [...]} stratified per dataset."""
    if languages:
        examples = [e for e in examples if e.get("language") in languages]
    if datasets:
        examples = [e for e in examples if e.get("dataset") in datasets]

    by_ds: dict[str, list[dict]] = {}
    for e in examples:
        by_ds.setdefault(e["dataset"], []).append(e)

    val, test = [], []
    for ds, items in sorted(by_ds.items()):
        items = sorted(items, key=lambda x: str(x["idx"]))
        rng = random.Random(f"{seed}:{ds}")
        rng.shuffle(items)
        if limit_per_dataset:
            items = items[:limit_per_dataset]
        n_val = max(1, int(round(len(items) * val_frac)))
        val.extend(items[:n_val])
        test.extend(items[n_val:])
    return {"val": val, "test": test}


def load_split(
    split_name: str,
    val_frac: float = 0.2,
    seed: int = 0,
    english_only: bool = True,
    limit_per_dataset: int | None = None,
) -> list[dict]:
    examples = load_raw()
    datasets = ENGLISH_DATASETS if english_only else None
    parts = split(
        examples,
        val_frac=val_frac,
        seed=seed,
        datasets=datasets,
        limit_per_dataset=limit_per_dataset,
    )
    return parts[split_name]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download + describe LongBench-Cite splits")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--all-languages", action="store_true")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    args = ap.parse_args()

    if args.download:
        download(force=True)

    raw = load_raw()
    parts = split(
        raw,
        val_frac=args.val_frac,
        seed=args.seed,
        datasets=None if args.all_languages else ENGLISH_DATASETS,
        limit_per_dataset=args.limit_per_dataset,
    )
    from collections import Counter

    for name, items in parts.items():
        print(f"{name}: {len(items)}  {dict(Counter(x['dataset'] for x in items))}")
    write_json(
        os.path.join(DATA_DIR, f"split_seed{args.seed}.json"),
        {"val": [e["idx"] for e in parts["val"]], "test": [e["idx"] for e in parts["test"]]},
    )
    print("wrote split index (idx lists) to data/")

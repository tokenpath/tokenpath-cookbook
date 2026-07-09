"""Cache raw TokenPath heatmaps for the native answer set — once.

The heatmap is a function of (document, question, answer) ONLY — it does not
depend on how we aggregate mass into citations. So we fetch every answer's
heatmap once and store the raw sparse response; every aggregation experiment
then recomputes citations offline (free, instant) from the cache and only pays
the judge. Turns a ~$8/15-min hill-climb iteration into ~$0.6 on a dev subset.

Stored gzipped per idx under results/exp1nat_heatmaps/<idx>.json.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import threading

from ..common import env
from ..common.io_utils import read_jsonl
from . import load_data, native_run

HM_DIR = os.path.join(native_run.RESULTS_DIR, "exp1nat_heatmaps")


def hm_path(idx) -> str:
    return os.path.join(HM_DIR, f"{idx}.json.gz")


def load_heatmap(idx) -> dict | None:
    p = hm_path(idx)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Cache raw heatmaps for the native answer set")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(HM_DIR, exist_ok=True)
    ex = {e["idx"]: e for e in load_data.load_split(args.split, seed=0, english_only=True,
                                                    limit_per_dataset=args.limit_per_dataset)}
    frozen = {r["idx"]: r["prediction"] for r in read_jsonl(native_run.FROZEN_PATH)}
    todo = [idx for idx in ex if idx in frozen and not os.path.exists(hm_path(idx))]
    print(f"heatmaps: {len(ex) - len(todo)} cached, {len(todo)} to fetch")

    tp = env.tokenpath_client()
    lock = threading.Lock()
    done = [0]

    def worker(idx):
        e = ex[idx]
        timed = tp.heatmap(e["context"], e["query"], frozen[idx])
        with gzip.open(hm_path(idx), "wt", encoding="utf-8") as f:
            json.dump(timed.value, f)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  {done[0]}/{len(todo)}")
        return {"idx": idx}

    from ..common.io_utils import parallel_append
    # write a tiny manifest as we go (also serves as a done-marker log)
    parallel_append(os.path.join(native_run.RESULTS_DIR, "exp1nat_heatmap_manifest.jsonl"),
                    todo, worker, workers=args.workers, desc="heatmaps")
    print("done. cached dir:", HM_DIR)


if __name__ == "__main__":
    main()

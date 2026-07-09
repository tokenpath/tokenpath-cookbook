"""Load WebCode: anti-memorization QA pairs + each provider's search results.

WebCode (Exa, 2026) is 317 {query, expected_answer} pairs drawn from
authoritative docs (GNU, W3C, IETF RFCs, Python/Rust/Go official docs), filtered
so two frontier models fail to answer from parametric memory over three
completions — i.e. the answer must be *retrieved*, not recalled. The queries were
dispatched to five providers (Exa, Brave, Perplexity, Parallel, Tavily) and
graded on groundedness / citation precision.

We do not redistribute WebCode or provider outputs. Point this loader at a JSONL
you assemble from Exa's release + your own provider API calls, one line per
(query, provider):

    {"qid": "webcode-001",
     "query": "...",
     "expected_answer": "...",
     "provider": "exa",
     "results": [{"url": "...", "text": "<snippet or page text>"}, ...]}

A tiny synthetic fixture (data/webcode/sample.jsonl) is written by --make-sample
so the harness runs end-to-end offline before you have the real data.
"""

from __future__ import annotations

import argparse
import os

from ..common.io_utils import ensure_dir, read_jsonl

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "webcode")
DEFAULT_PATH = os.path.join(DATA_DIR, "webcode.jsonl")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample.jsonl")

PROVIDERS = ["exa", "brave", "perplexity", "parallel", "tavily"]


def load(path: str = DEFAULT_PATH) -> list[dict]:
    rows = read_jsonl(path)
    for r in rows:
        assert {"qid", "query", "expected_answer", "provider", "results"} <= set(r), (
            f"row missing required fields: {set(r)}"
        )
    return rows


def by_provider(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["provider"], []).append(r)
    return out


def make_sample(path: str = SAMPLE_PATH) -> str:
    """A 2-query x 2-provider fixture for smoke-testing the pipeline offline."""
    import json

    ensure_dir(DATA_DIR)
    rows = [
        {
            "qid": "webcode-sample-001",
            "query": "What signal does POSIX raise when a process writes to a closed pipe?",
            "expected_answer": "SIGPIPE",
            "provider": "exa",
            "results": [
                {"url": "https://pubs.opengroup.org/x", "text":
                 "If a process attempts to write to a pipe that has no readers, the "
                 "SIGPIPE signal is generated for the writing process."},
                {"url": "https://example.com/unrelated", "text":
                 "The tar utility creates archive files from a set of directories."},
                {"url": "https://example.com/noise", "text":
                 "Cron schedules jobs at fixed times, dates, or intervals."},
            ],
        },
        {
            "qid": "webcode-sample-001",
            "query": "What signal does POSIX raise when a process writes to a closed pipe?",
            "expected_answer": "SIGPIPE",
            "provider": "brave",
            "results": [
                {"url": "https://man7.org/x", "text":
                 "Writing to a pipe whose read end is closed delivers SIGPIPE to the writer."},
                {"url": "https://example.com/blog", "text":
                 "A history of Unix pipes and their design philosophy."},
            ],
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()
    if args.make_sample:
        print("wrote", make_sample())
    else:
        rows = load(args.path)
        bp = by_provider(rows)
        print({p: len(v) for p, v in bp.items()}, "total", len(rows))

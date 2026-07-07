"""Small IO helpers: JSONL append with idx-dedup, JSON dump, and a file cache.

The harness is checkpointed at every expensive step (frozen answers, per-method
citations, judge scores) so a crashed or rate-limited run resumes instead of
paying for the same tokens twice. Cache keys are explicit and content-derived so
changing an input invalidates the right entries.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, record: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_done_ids(path: str, key: str = "idx") -> set:
    return {r[key] for r in read_jsonl(path) if key in r}


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cache_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def dedup_by(records: Iterable[dict], key: str = "idx") -> list[dict]:
    seen: dict = {}
    for r in records:
        seen[r[key]] = r
    return list(seen.values())

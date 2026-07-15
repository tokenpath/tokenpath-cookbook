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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable


_SIGNATURE_NOT_PROVIDED = object()


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


def is_error_record(record: dict) -> bool:
    """Whether a cached record represents a failed attempt, not completed work."""
    return "error" in (record.get("extra") or {})


def record_cache_signature(record: dict) -> Any:
    """Return the producer signature stamped into a cached record, if any."""
    return (record.get("extra") or {}).get("cache_signature")


def load_done_ids(
    path: str,
    key: str = "idx",
    expected_signature: Any = _SIGNATURE_NOT_PROVIDED,
) -> set:
    """Return successfully completed IDs using the latest attempt for each ID.

    Failed attempts remain in JSONL as diagnostics but must be retryable. Using
    last-write-wins also lets a later successful retry supersede the old error.
    """
    records = [r for r in read_jsonl(path) if key in r]
    latest = dedup_by(records, key=key)
    return {
        r[key]
        for r in latest
        if not is_error_record(r)
        and (
            expected_signature is _SIGNATURE_NOT_PROVIDED
            or record_cache_signature(r) == expected_signature
        )
    }


def parallel_append(
    path: str,
    items: list,
    worker: Callable[[Any], dict | None],
    workers: int = 8,
    desc: str | None = None,
) -> None:
    """Run `worker(item) -> record` over items concurrently, appending each
    record to `path` as it finishes. Every LLM/API call the harness makes per
    item is independent, so the wall-clock is bounded by the slowest item, not
    the sum. Writes are serialized by a lock; the file stays valid JSONL and
    idx-dedup on resume still works. `workers=1` runs sequentially (use for
    local, non-thread-safe work like torch inference)."""
    from tqdm import tqdm

    lock = threading.Lock()

    def run(item):
        rec = worker(item)
        if rec is not None:
            with lock:
                append_jsonl(path, rec)
        return rec

    if workers <= 1:
        for it in tqdm(items, desc=desc):
            run(it)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run, it) for it in items]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=desc):
            pass


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


def read_jsonl_latest(path: str, key: str = "idx") -> list[dict]:
    """Read an append-only JSONL cache with the newest record winning per key."""
    return dedup_by((r for r in read_jsonl(path) if key in r), key=key)

"""Average citation length, matching LongCite/LongBench-Cite/cnt_citation_len.py.

LongCite reports citation length as GLM-4 tokens per citation snippet. We use the
same tokenizer so the "citation length" column is comparable to published rows.
The tokenizer download is guarded: if transformers/the model is unavailable, we
fall back to a whitespace word count and label the column accordingly.
"""

from __future__ import annotations

_tokenizer = None
_backend = None


def _load():
    global _tokenizer, _backend
    if _backend is not None:
        return
    try:
        from transformers import AutoTokenizer  # type: ignore

        _tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/glm-4-9b-chat", trust_remote_code=True
        )
        _backend = "glm4"
    except Exception:
        _tokenizer = None
        _backend = "words"


def token_len(text: str) -> int:
    _load()
    if _backend == "glm4":
        return len(_tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def backend() -> str:
    _load()
    return _backend  # "glm4" (comparable to published) or "words" (fallback)


def mean_citation_len(records: list[dict]) -> tuple[float, int]:
    """Mean citation length and citation count over judged records."""
    total, n = 0, 0
    for js in records:
        for sc in js.get("statements", []):
            for c in sc.get("citation", []):
                total += token_len(c["cite"])
                n += 1
    return (total / n if n else 0.0), n

# TokenPath Cookbook

Runnable examples for [TokenPath](https://tokenpath.ai) — token-level attribution
for LLM output. One API call after generation maps every span of a model's
answer back to the exact source spans that produced it, measured from model
attention. Not "this answer cites this chunk" — *these 6 words came from those
9 words*.

```bash
curl https://api.tokenpath.ai/v1/attributions \
  -H "Authorization: Bearer $TOKENPATH_API_KEY" \
  -d '{
    "document": "In Q3, Northwind's revenue grew 18%…",
    "question": "How fast did revenue grow?",
    "answer":   "Revenue grew 18% year over year.",
    "spans":    [[13, 16]]
  }'
```

Works with output from **any** model — nothing about your generation stack
changes.

## Notebooks

| Notebook | What it shows |
| --- | --- |
| [01 — Grounding gate](notebooks/01-grounding-gate.ipynb) | Block unsupported claims before they ship. Drop-in gate + LangChain & LlamaIndex wiring, and the two-stage attribution → verification pattern. |
| [02 — Citation highlighting](notebooks/02-citation-highlighting.ipynb) | Perplexity-style citations whose highlights land on the exact words, not the chunk. Inline + standalone HTML, and the bracket-and-link pattern. |
| [03 — Groundedness eval metric](notebooks/03-groundedness-eval.ipynb) | Groundedness as a deterministic scalar for eval harnesses and reward signals — no LLM judge. |
| [04 — Heatmap visualization](notebooks/04-heatmap-visualization.ipynb) | The raw answer-token × document-token attention matrix: densify, plot, and roll up. |

Open any of them in Colab:
[01](https://colab.research.google.com/github/tokenpath/cookbook/blob/main/notebooks/01-grounding-gate.ipynb) ·
[02](https://colab.research.google.com/github/tokenpath/cookbook/blob/main/notebooks/02-citation-highlighting.ipynb) ·
[03](https://colab.research.google.com/github/tokenpath/cookbook/blob/main/notebooks/03-groundedness-eval.ipynb) ·
[04](https://colab.research.google.com/github/tokenpath/cookbook/blob/main/notebooks/04-heatmap-visualization.ipynb)

## Running locally

```bash
git clone https://github.com/tokenpath/cookbook.git
cd cookbook
pip install -r requirements.txt
export TOKENPATH_API_KEY=...   # free key: https://platform.tokenpath.ai
jupyter lab notebooks/
```

Every notebook needs a `TOKENPATH_API_KEY` — sign up at
[platform.tokenpath.ai](https://platform.tokenpath.ai) for **10M free attributed
tokens, no card required**. After that it's $1 per 1M attributed tokens, pay as
you go.

## Getting help

- 🐛 **Bugs & broken examples** → [open an Issue](https://github.com/tokenpath/cookbook/issues/new/choose).
  Fastest way to get something fixed — include the notebook name and the
  `request_id` from any API error.
- 💬 **"How do I…?" questions** → [start a Discussion](https://github.com/tokenpath/cookbook/discussions).
  Integration questions, use-case design, "is TokenPath right for X" — anything
  that isn't a bug.
- 📚 **API reference** → [docs.tokenpath.ai](https://docs.tokenpath.ai)
- ✉️ **Private/billing questions** → support@tokenpath.ai

## Contributing

Recipes are welcome. Open an issue describing the use case first (so we can
point you at prior art), then PR a notebook that follows the pattern here:
self-contained, runnable top-to-bottom with only `TOKENPATH_API_KEY` set, heavy
dependencies behind a guarded cell.

## License

[MIT](LICENSE)

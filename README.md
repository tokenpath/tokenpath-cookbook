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

Every recipe here is something we've built and run for real — executed against
the live API, outputs included. The collection grows as we ship.

| Notebook | What it shows |
| --- | --- |
| [Citation highlighting](notebooks/citation-highlighting.ipynb) | Perplexity-style citations whose highlights land on the exact words, not the chunk. Inline + standalone HTML, and the bracket-and-link pattern. |
| [Structured output attribution](notebooks/structured-output-attribution.ipynb) | Trace every extracted JSON field back to its exact source span — including disambiguating repeated values that naive string search maps to the wrong occurrence. |
| [Transcript Q&A attribution](notebooks/transcript-qna-attribution.ipynb) | Meeting notes and transcript answers where every claim links back to the speaker, timestamp, and exact words behind it. |
| [Heatmap visualization](notebooks/heatmap-visualization.ipynb) | The raw answer-token × document-token attention matrix: densify, plot, and roll up. |

Open in Colab:
[citation highlighting](https://colab.research.google.com/github/tokenpath/tokenpath-cookbook/blob/main/notebooks/citation-highlighting.ipynb) ·
[structured output](https://colab.research.google.com/github/tokenpath/tokenpath-cookbook/blob/main/notebooks/structured-output-attribution.ipynb) ·
[transcript Q&A](https://colab.research.google.com/github/tokenpath/tokenpath-cookbook/blob/main/notebooks/transcript-qna-attribution.ipynb) ·
[heatmap visualization](https://colab.research.google.com/github/tokenpath/tokenpath-cookbook/blob/main/notebooks/heatmap-visualization.ipynb)

### On the roadmap

Tracked as issues — comment or 👍 to prioritize:

- [PDF attribution](https://github.com/tokenpath/tokenpath-cookbook/issues/1) — cite back into the PDF itself, page + highlight
- [Cited search agent](https://github.com/tokenpath/tokenpath-cookbook/issues/4) — the tokenpath.ai search demo, end to end

## Running locally

```bash
git clone https://github.com/tokenpath/tokenpath-cookbook.git
cd tokenpath-cookbook
pip install -r requirements.txt
export TOKENPATH_API_KEY=...   # free key: https://platform.tokenpath.ai
jupyter lab notebooks/
```

Every notebook needs a `TOKENPATH_API_KEY` — sign up at
[platform.tokenpath.ai](https://platform.tokenpath.ai) for **10M free attributed
tokens, no card required**. After that it's $1 per 1M attributed tokens, pay as
you go.

## Getting help

- 🐛 **Bugs & broken examples** → [open an Issue](https://github.com/tokenpath/tokenpath-cookbook/issues/new/choose).
  Fastest way to get something fixed — include the notebook name and the
  `request_id` from any API error.
- 💬 **"How do I…?" questions** → [start a Discussion](https://github.com/tokenpath/tokenpath-cookbook/discussions).
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

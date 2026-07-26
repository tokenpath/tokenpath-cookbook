# Exp 4 — within-model: attention vs the model's own words

Model under test: `local/qwen3.5-9b` (one model writes the answer, cites it, and is the model whose attention is read).
Split: `test` (n=480) · judge: `google/gemini-2.5-flash` · attention mass threshold: 0.1 (coarse val sweep, n=120, interior optimum).

F1 is LongCite's reported average (excludes multifieldqa).

## Arm 1 — frozen answer

| condition | n | F1 (all) | F1 (well-formed) | F1 paired | attention, same examples | gap (paired) | format ok | truncated |
|---|---|---|---|---|---|---|---|---|
| A: prompted | 480 | 0.501 | 0.549 | 0.549 | 0.739 | **+0.191** | 92% | 2% |
| A: prompted_thinking | 480 | 0.603 | 0.644 | 0.644 | 0.744 | **+0.100** | 94% | 5% |
| B: attention (all) | 480 | 0.738 | — | — | — | — | n/a | n/a |

## Arm 2 — single-pass inline vs second pass

| condition | n | F1 (all) | F1 (well-formed) | F1 paired | attention, same examples | gap (paired) | format ok | truncated |
|---|---|---|---|---|---|---|---|---|
| A: inline | 480 | 0.264 | 0.263 | 0.263 | 0.805 | **+0.542** | 97% | 3% |
| A: prompted_2pass | 480 | 0.561 | 0.624 | 0.624 | 0.798 | **+0.174** | 90% | 1% |
| B: attention (all) | 480 | 0.803 | — | — | — | — | n/a | n/a |

## Cost and latency (same weights, same GPU)

| row | p50 latency | $/query |
|---|---|---|
| exp4/prompted | 3.55s | $0.01126 |
| exp4/prompted_thinking | 53.06s | $0.07256 |
| exp4/tokenpath | 1.20s | $0.01269 |
| exp4inline/inline | 4.26s | $0.00606 |
| exp4inline/prompted_2pass | 4.63s | $0.00947 |
| exp4inline/tokenpath | 1.24s | $0.01269 |

#!/usr/bin/env bash
# One-command repro per table. Run from the repo root:
#   TOKENPATH_API_KEY=... OPENROUTER_API_KEY=... bash benchmarking/scripts/run_all.sh
#
# Optional: ANTHROPIC_API_KEY=... enables the Citations API baseline in Exp 1.
# Use SMOKE=1 to run a tiny slice end-to-end (a few examples per dataset) to
# verify everything is wired before spending on the full run.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

: "${TOKENPATH_API_KEY:?set TOKENPATH_API_KEY}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  LIM=(--limit-per-dataset 3); TLIM=(--limit-per-dataset 3)
  echo ">> SMOKE mode: tiny slice"
else
  LIM=(); TLIM=(--limit-per-dataset 5)   # tuning uses a small val sample by default
fi

echo "== Experiment 1: LongBench-Cite =="
python -m benchmarking.exp1_longbench_cite.load_data --download
# 1. tune TokenPath threshold on VAL only
python -m benchmarking.exp1_longbench_cite.tune_threshold "${TLIM[@]}"
# 2. run all methods on TEST (threshold read from exp1_threshold.json)
python -m benchmarking.exp1_longbench_cite.run --split test "${LIM[@]}"
# 3. (optional) re-judge anchor generations with our judge, if you have them:
#    python -m benchmarking.exp1_longbench_cite.rejudge_anchors \
#        --preds path/to/LongCite-8b-preds.json --name longcite-8b
# 4. render the table
python -m benchmarking.exp1_longbench_cite.make_table

echo "== Experiment 2: WebCode citation-precision filter =="
# Assemble benchmarking/data/webcode/webcode.jsonl from Exa's WebCode release +
# your provider API calls (schema in exp2_webcode/load_data.py). To smoke-test
# the wiring offline first:  python -m benchmarking.exp2_webcode.load_data --make-sample
DATA="benchmarking/data/webcode/webcode.jsonl"
[[ "${SMOKE:-0}" == "1" ]] && DATA="benchmarking/data/webcode/sample.jsonl" && \
  python -m benchmarking.exp2_webcode.load_data --make-sample
python -m benchmarking.exp2_webcode.run --data "$DATA"
python -m benchmarking.exp2_webcode.make_chart

echo "== Experiment 3: memorization detection (rides on Exp 2 data) =="
python -m benchmarking.exp3_memorization.run

echo "== done. Tables + figures in benchmarking/results/ =="

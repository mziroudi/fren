#!/usr/bin/env bash
# Collect routing traces for both tasks (greedy decoding, 200 generated tokens per input).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf}
mkdir -p results/raw
for task in code_explain news_summary; do
  ./build/route_trace -m "$MODEL" -i data/$task.jsonl -n 200 -t 4 \
    -o results/raw/trace_$task.tsv -g results/raw/gens_$task.jsonl 2> results/raw/log_$task.txt
done

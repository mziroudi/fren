#!/usr/bin/env bash
# Reproduce every number in results/WEEK1.md.
# Needs: cmake, a C++17 compiler, curl, Python with requirements.txt. ~8 GB disk.
# The capture step takes hours on a 4-core CPU (about 7 h on the iMac in results/hardware.md).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
if [ ! -f "$MODEL" ]; then
  mkdir -p models
  # 7,359,942,656 bytes, Apache-2.0
  curl -L -o "$MODEL" https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/resolve/main/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
fi
./capture/build.sh
./capture/run_all.sh             # -> results/raw/
python3 analysis/to_parquet.py   # -> results/routing_trace.parquet, results/routing.parquet (+ sanity checks)
python3 analysis/coverage.py     # -> results/coverage.png, coverage_by_layer.csv, coverage_summary.json
python3 analysis/cache_sim.py    # -> results/cache_sim.json

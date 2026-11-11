#!/usr/bin/env bash
# Fetch the pinned llama.cpp commit and build route_trace (CPU only).
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_SHA=d2b3eb9b6030d1cb6a2c05fa07cc71b85da69f34
if [ ! -f vendor/llama.cpp/CMakeLists.txt ]; then
  mkdir -p vendor/llama.cpp
  curl -sL "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_SHA}.tar.gz" \
    | tar xz -C vendor/llama.cpp --strip-components=1
fi
cmake -S capture -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target route_trace -j 4
echo "built: build/route_trace"

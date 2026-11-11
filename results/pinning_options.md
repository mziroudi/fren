# Pinning options (feasibility note, no implementation)

Checked against llama.cpp master `d2b3eb9b6030d1cb6a2c05fa07cc71b85da69f34` (2026-11-11).

## What the source says

- **Experts are stacked, one tensor per layer per matrix.** `blk.N.ffn_{gate,up,down}_exps`
  have shape `{n_embd, n_ff, n_expert}` (`src/models/olmoe.cpp`). Expert is the outermost
  dimension, so **each expert is one contiguous byte range** inside each of the three
  tensors. For OLMoE Q8_0 (n_embd 2048, n_ff 1024, 34 bytes per 32 weights) that is
  3 × 2048 × 1024 × 34/32 ≈ 6.7 MB per expert per layer (computed, not measured).
- **`--override-tensor` / `--cpu-moe` / `--n-cpu-moe`** (`common/arg.cpp`) match tensor
  names, so they place a whole layer's experts at once. No per-expert placement.
- **New: llama.cpp already has a per-expert GPU cache.** `--moe-cache-mib N`
  (`src/llama-moe-cache.{h,cpp}`) keeps "recently and repeatedly used experts of
  host-resident MoE layers in a device buffer". The policy is a combined recency/frequency
  score with a 128-token half-life, log2 frequency classes with one LRU list each, and slots
  shared by the layers of a group. On a miss it either uploads the expert or
  (`--moe-cache-cpu-share`) computes it on the CPU. There is no API to seed or pin
  entries. It needs a GPU backend, so it does not help this iMac (2 GB AMD, CPU-only build).
- **Prefetch helpers exist.** `llama_prefetch` / `llama_prefetch_rows` (`src/llama-mmap.cpp`,
  `src/llama-impl.cpp`) issue `madvise(MADV_WILLNEED)` on byte ranges. Today they are used
  for per-layer embeddings, not experts. That is the natural hook for the later "prefetch"
  phase.

## Options

| Option | What it does | Tiers | Effort | Risk |
|---|---|---|---|---|
| (a) per-layer `--override-tensor` | Put whole layers' experts in VRAM or on CPU. | VRAM, layer-granular only | none (exists) | Cannot express a hot set inside a layer, so it doesn't use our result. It's the baseline, not our product. |
| (b) helper process locks hot-expert byte ranges | `mmap` the same GGUF read-only, `mlock` (or touch periodically) the byte ranges of hot experts so the OS keeps them in the page cache; llama.cpp unchanged. | RAM only | small: ~200 lines (read GGUF tensor offsets, compute expert slabs, mlock) | `mlock` limits (`ulimit -l`, macOS needs root or entitlement for large locks); must confirm locked pages in one process are the same page-cache pages llama.cpp maps (true for shared file mappings on Linux; to be verified on macOS and Windows). Value depends on the RAM tier beating the page cache's own policy (see `cache_sim.json`). |
| (c) seed or pin llama.cpp's MoE cache | Patch `llama-moe-cache.cpp` so a profile file marks experts as pinned (never evicted) and preloads them at start. | VRAM (GPU builds) | medium: a focused patch to one file plus a CLI flag; upstreamable | Requires a GPU; must beat the existing CRF policy, which already adapts online. Patch has to track an actively changing file. |
| (c') repack GGUF into hot/cold tensors | Rewrite the GGUF so hot experts are separate tensors, then place them with `--override-tensor`. | VRAM + RAM | large: repacker plus a routing remap patch in `build_moe_ffn` | Mostly superseded by the MoE cache above; a remap patch is close to "writing an engine". Not recommended. |

## Assessment

- Per-expert placement **is possible today** but only dynamically (the GPU MoE cache), and
  only on GPU builds. Static per-expert pinning does not exist.
- (b) is feasible without touching llama.cpp and targets machines like this one (RAM tier,
  model bigger than RAM). It only pays off if a static hot set beats the page cache's LRU-like
  eviction. The simulation answers that.
- (c) is the realistic VRAM-tier path and is now much smaller than the original plan, because
  llama.cpp already did the hard part. Our contribution would be the profile, seeding and
  pinning, not the cache.

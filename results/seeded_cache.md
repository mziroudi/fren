# Does a task profile help an adaptive expert cache? (SIMULATION)

Follow-up to `results/WEEK1.md`. Week 1 showed that a static hot set barely beats LRU and
loses to a policy like llama.cpp's own MoE cache. This tests whether the profile helps **on
top of** those adaptive caches. Every number comes from `python3 analysis/seeded_cache_sim.py`
(output: `results/seeded_cache.json`) on the week-1 trace. No new model run was needed.

**Short answer:**
- **The profile helps a little, and only in two places:**
  - pinning half the budget next to the OS page cache: 5–10% fewer misses;
  - warm-starting llama.cpp's MoE cache in a fresh process: 2–13% fewer misses over 200
    tokens, and 14–57% fewer in the first 16 tokens.
- **It does nothing for a long-running llama.cpp server.**
- **Even the best case** still leaves about 18 of each token's 128 expert selections cold
  at a 50% budget.

## What was simulated

Profile: task A (code) half A, generated tokens. Replay: task A half B (40 held-out inputs,
8,000 generated tokens). Metric: cold experts per generated token, out of 128 selections
(16 layers × top-8).

Two adaptive caches:
- **LRU** stands in for the OS page cache (RAM tier). Prompt tokens update it, because the
  prompt reads the same mmapped expert pages.
- **CRF** is a simplified version of llama.cpp's `--moe-cache-mib` policy (VRAM tier).
  Prompt tokens do *not* update it: llama.cpp skips its MoE cache for ubatches over 32 tokens
  (`src/llama-moe-cache.cpp`, `max_batch = 32`).

Three ways of using the profile:

| Variant | Meaning |
|---|---|
| plain | no profile; the cache adapts on its own |
| seeded | the cache starts filled with the profile's top experts (CRF: scores set from profile frequencies), then adapts |
| pinned | half the budget is locked to the profile's top experts; the other half adapts over the remaining experts. For the RAM tier this is option (b) in `pinning_options.md` (helper process locks hot experts) plus the page cache |

Two scenarios:
- **Fresh:** the cache resets before every input (a new process per request, e.g. `llama-cli`).
- **Long-running:** the cache carries over across inputs after replaying half A first (a
  server that has been up a while).

## Results: cold experts per generated token (lower is better)

Mean over all 200 generated tokens; in brackets, the mean over the first 16 generated tokens
of each input.

### RAM tier (LRU = page-cache stand-in), fresh scenario

| Budget per layer | Static hot set (week 1) | LRU plain | LRU seeded | **LRU + pinned half** |
|---|---|---|---|---|
| 10% (6) | 81.1 | 112.7 | 112.7 | 91.8 |
| 20% (13) | 59.4 | 61.5 | 61.5 | **58.7** (−5% vs plain) |
| 30% (19) | 45.2 | 46.6 | 46.5 | **42.9** (−8%) |
| 50% (32) | 22.2 | 24.0 | 24.0 | **21.6** (−10%) |

- **Seeding** the page cache does nothing, because the first prompt fills the cache anyway.
  The long-running scenario gives the same numbers.
- **Pinning half the budget** cuts misses by 5–10% against LRU alone at 20–50% budgets. It is
  also 1–5% better than the static set alone. At 10% the budget is smaller than top-8, and a
  plain static set is best.

### VRAM tier (CRF = llama.cpp MoE cache policy)

| Budget | Fresh: plain | **Fresh: seeded** | Fresh: pinned half | Long-running: plain | Long-running: seeded | Long-running: pinned half |
|---|---|---|---|---|---|---|
| 10% | 78.1 [84.4] | 77.7 [80.7] | 77.3 [81.0] | 77.8 | 77.8 | 77.1 |
| 20% | 52.8 [63.5] | **51.6 [54.7]** | 52.5 [58.3] | 51.9 | 51.9 | 52.1 |
| 30% | 40.1 [54.4] | **38.5 [41.1]** | 39.4 [46.3] | 38.9 | 38.9 | 38.9 |
| 50% | 21.1 [43.1] | **18.5 [18.4]** | 19.7 [28.6] | 19.0 | 19.0 | 18.9 |

- **Fresh process, seeded:**
  - Over the whole 200-token generation: 2–13% fewer misses (−4% at 30%, −13% at 50%).
  - In the first 16 tokens: 14–57% fewer misses (−24% at 30%, −57% at 50%).
  - In effect, seeding removes the warm-up cost.
- **Long-running server:** the profile adds nothing. The adaptive cache already settled on the
  same experts (seeded = plain; pinned within 1%).

## What this means

1. **No result here reverses the week-1 NO-GO.** The best case, a fresh llama.cpp process
   with a 50% cache seeded from a profile, still has about 18 cold experts per token. On a
   machine where cold means "read from SSD", every token still waits on the SSD.
2. **The profile is a warm-start, not a placement plan.** Its value is highest for short
   sessions on GPU builds (llama.cpp's MoE cache). For long-running servers it adds nothing.
3. **For the RAM tier** (machines like this iMac), "lock half the budget plus page cache" is
   worth 5–10% fewer misses. That would need the mlock helper, option (b), and only pays off
   if those 5–10% are real SSD reads.
4. **Smallest real next step, if any:** a `--moe-cache-seed <profile>` patch to llama.cpp's
   MoE cache (one file, upstreamable). It needs a GPU machine to benchmark; this iMac can't
   run it.

## Limits

- **This is a simulation.** The same simplifications as `analysis/cache_sim.py` apply:
  - one expert = one unit, with a separate budget per layer;
  - strict LRU stands in for the page cache;
  - CRF keeps llama.cpp's scoring idea but not its frequency classes or slots shared across
    layers.
- **The CRF seed is a guess.** Initial scores are profile frequency × the steady-state score
  of an expert used every token; that number is not taken from llama.cpp.
- **"Pinned half" is one split.** Other splits between pinned and adaptive were not tried.
- **Narrow data.** One model (OLMoE-1B-7B Q8_0), task A only, 40 held-out inputs.

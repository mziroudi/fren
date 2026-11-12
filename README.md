# fren

**Status: stopped (2026-11-12) after the week-1 kill test returned NO-GO.**

fren asked whether big Mixture-of-Experts models could run faster on normal PCs by profiling
which experts a task uses and pinning those experts in fast memory (see `CLAUDE.md`).

## What we found (OLMoE-1B-7B-0125-Instruct Q8_0, llama.cpp)

- **Routing is not skewed enough.** Covering 80% of a code task's expert selections takes
  about 45% of the experts in the median layer; the go criterion was 20–30%. Every token
  makes 128 expert selections, and even with 50% of experts pinned, about 22 of them per
  token are still cold.
- **The hot set is very stable within a task and very different between tasks.** A set
  built on 40 inputs was within 0.1 point of the best possible set on 40 others. The code
  task's hot set was no better than random on news summarization.
- **Existing caching already does about as well (simulation):**
  - A static hot set beats an LRU page-cache model by only 3–7%.
  - A simplified model of llama.cpp's own per-expert GPU cache (`--moe-cache-mib`) matches
    or beats it.
  - Using the profile to seed or partly pin those adaptive caches gains 2–13%, mostly while
    a fresh process warms up.

Details: `results/WEEK1.md` and `results/seeded_cache.md`. To reproduce everything (about
7 hours on a 4-core CPU, mostly the routing capture):

```bash
./reproduce.sh
```

## What is reusable

- `capture/route_trace.cpp` logs per-token, per-layer expert choices from any llama.cpp MoE
  model, using the `cb_eval` callback on `ffn_moe_topk`.
- `analysis/` has scripts for coverage, held-out stability, task specificity, misses per
  token, and cache-policy simulations, all working on that trace.

## Open questions not tested

- Models with many fine-grained experts (e.g. Qwen3-30B-A3B, 128 experts) may be more
  skewed.
- A small llama.cpp patch that fills the MoE cache from a profile at startup could help
  short GPU sessions. It needs a machine with a real GPU to measure.

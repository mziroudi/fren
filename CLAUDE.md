# fren

## What this project is
An open-source tool, built on top of llama.cpp, that makes big Mixture-of-Experts (MoE)
models run faster on normal PCs when the model does not fit in VRAM + RAM.

1. **Profile**: run the model on a task and log which experts the router picks.
2. **Pin**: keep the most-used experts in fast memory, leave the rest on SSD.
3. **Prefetch** (later): predict the next experts and load them early.

Why it could work: each generated token reads the weights of the experts the router
selects. Reads from SSD are far slower than reads from RAM or VRAM. If a task keeps
selecting a small, stable subset of experts, keeping that subset in fast memory removes
most slow reads.

## What we already know (verify each against current sources before relying on it)
- llama.cpp stores all experts of a layer in one stacked tensor (`blk.N.ffn_up_exps`,
  `ffn_gate_exps`, `ffn_down_exps`). `--override-tensor` works per tensor, so it can move
  a whole layer's experts but NOT individual experts. Per-expert pinning needs another
  mechanism (see step 7).
- llama.cpp mmaps the model, so the OS page cache already keeps recently used experts in
  RAM. That is the baseline we have to beat, not "everything on SSD".
- A token needs ALL of its top-k experts in every layer. "80% of selections covered" still
  means most tokens hit at least one cold expert. Selection coverage alone overstates the
  speedup, so we also measure misses per token.
- Prompt processing runs many tokens per batch and touches nearly every expert. The benefit,
  if any, is in generation. Analyze prompt tokens and generated tokens separately.
- Prior research did popularity-based expert placement (Fiddler, MoE-Infinity, EdgeMoE,
  Pre-gated MoE). Read their routing-skew results before interpreting ours.
- Routing is a property of model + input, not of hardware. The profile can be collected on
  any machine that fits the model.

## Rules for this repo
- We do NOT build a new inference engine or a chat app. Build on llama.cpp / existing tools.
- One model and one task first. Generalize later.
- Every number in a results file comes from a script in this repo that reproduces it.
  Fixed seeds, greedy decoding, pinned package versions.
- Never estimate or fill in a number that was not measured. If a step could not run, say so
  in the results file and say why.
- The owner has ~10 hours/week. Prefer the smallest step that produces a real number.
- Keep everything runnable on consumer hardware.
- Verify model size and license before any download. Ask the owner before downloading
  more than 20 GB.

## Current phase: WEEK 1, the kill test
Question: **for one task, does a small, stable set of experts handle most of the routing,
and would pinning that set beat what the OS page cache already does?**

### Steps
1. **Hardware.** Record OS, CPU, RAM, GPU + VRAM, free disk, SSD type in
   `results/hardware.md`. State plainly whether this is the owner's PC or a cloud
   container. If it is a container, the file describes where the profile ran, not the
   target machine; ask the owner for their PC specs for the target.
2. **Model.** Pick the MoE that fits this machine and write the reason in
   `results/hardware.md`. Small machine: OLMoE-1B-7B via Hugging Face transformers on CPU.
   More memory: Qwen3-30B-A3B quantized via llama.cpp. Record expert count, top-k and
   layer count. Results are specific to the model profiled; say so in the write-up.
3. **Data.** Task A: 50–100 inputs for one task (e.g. explain a code file). Task B: 50–100
   inputs for a clearly different task (e.g. summarize a contract or news article). Use
   permissively licensed sources and record them in `data/README.md`.
4. **Log routing as a trace, not only counts.** For each input, run the prompt and generate
   about 200 new tokens. For every token and every MoE layer record the selected experts.
   Save `results/routing_trace.parquet` with columns
   `task, input_id, token_pos, phase (prompt|gen), layer, expert`.
   Derive `results/routing.parquet` (task, layer, expert, count) from it.
   - transformers route: forward hooks on each MoE block's router, or
     `output_router_logits=True` where supported.
   - llama.cpp route: the `cb_eval` callback, capturing the top-k node per layer. Read the
     current `build_moe_ffn` source for the node names.
   Sanity check: every token has exactly top-k experts per layer, and counts sum correctly.
5. **Analyze** in `analysis/coverage.py`, generated tokens reported separately from prompt:
   - Per layer: % of experts that covers 50%, 80%, 95% of selections. Chart in
     `results/coverage.png`. Include the uniform-routing line for reference.
   - Stability: split task A inputs in two halves. Build the hot set from half A, measure
     its coverage on half B.
   - Task specificity: coverage of task B by task A's hot set, and the overlap of the sets.
   - Misses per token: for budgets of 10/20/30/50% of experts per layer, hot set from
     half A, measured on half B: average cold experts per generated token (summed over
     layers) and the share of tokens with zero cold experts.
6. **Page-cache baseline** in `analysis/cache_sim.py`: replay the half-B trace through an
   LRU cache of the same expert budget, and report its misses per generated token next to
   the static hot set's. This is a simulation; label it as one.
7. **Pinning feasibility note**, `results/pinning_options.md`, one page, no implementation.
   Confirm from the llama.cpp source whether per-expert placement is possible today, then
   assess: (a) per-layer `--override-tensor` only, (b) locking the hot experts' byte ranges
   of the GGUF in the page cache from a helper process (no llama.cpp change, RAM tier
   only), (c) repacking the GGUF into hot/cold expert tensors plus a llama.cpp patch
   (needed for the VRAM tier). Give effort and risk for each.
8. **Write `results/WEEK1.md`**: the numbers, the charts, the limits of the test, and a
   clear GO / NO-GO with the reasoning.

### Decision rule
- **GO**: on held-out generated tokens, about 20–30% of experts cover 80%+ of selections,
  the hot set from half A holds up on half B, AND at least one option in step 7 is
  feasible without writing an inference engine.
- **NO-GO**: usage is close to uniform, or the hot set changes a lot between halves.
  Then try another task or model, or stop.
- The LRU comparison decides where the value is. If LRU matches the static hot set, the RAM
  tier adds little beyond warm-up and only the VRAM tier is worth building. Say this
  plainly in WEEK1.md.

### How to work
- Do the steps in order and commit after each one that produces a file.
- Stop and ask the owner before: any download over 20 GB, changing the model or task after
  results exist, or starting anything under "Later phases".
- Final message: the GO / NO-GO, the three or four numbers behind it, what could not be
  measured, and the command that reproduces everything.

## Later phases (do not start yet)
- Weeks 2–3: CLI `fren profile` + `fren pin`, using the option chosen in step 7.
- Week 4: before/after tokens-per-second benchmark on a model that does not fit in memory,
  reproducible script, compared against plain mmap.
- Week 5: README, demo video, one clear chart.
- Week 6: public launch.

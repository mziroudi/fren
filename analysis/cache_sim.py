"""SIMULATION: static hot set vs an LRU cache of the same size (a stand-in for the OS page cache).

This is a simulation, not a measurement of llama.cpp. It replays the routing trace of
task A half B (held-out inputs) through two policies with the same per-layer budget of b
experts, and counts how many selected experts of each generated token are not in fast
memory ("misses").

  static : hot set = top-b experts per layer from task A half A generated tokens. Never changes.
  lru    : per-layer LRU cache holding b experts. Every expert use (prompt and generated
           tokens, in order) loads it if missing and marks it most recently used.
           Two warm-up variants:
             cold  - cache starts empty at the first half-B input
             warm  - cache first replays half A (steady state, like a long-running server)
  lru_gen_only : like lru/warm, but only generated tokens touch the cache. This bounds the
           case where prompt processing would not evict generation experts.
  crf    : a simplified copy of the policy of llama.cpp's own per-expert GPU cache
           (--moe-cache-mib, src/llama-moe-cache.cpp): recency + frequency score with a
           128-token half-life. Same warm / gen-only variants.

Simplifications (say them when quoting the numbers):
  - Real page cache works on 16 KB pages (macOS) of the GGUF and is global across layers and
    all other processes; here the unit is one whole expert and each layer has its own budget.
  - macOS does not use strict LRU. LRU is the textbook approximation.
  - Within a token, the order the 8 experts are touched does not change hit/miss counts
    unless b < 8; at b = 6 (10%) they are taken in router order.

Writes results/cache_sim.json and prints it.
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TASK_A = "code_explain"
BUDGETS = [0.10, 0.20, 0.30, 0.50]


def token_stream(df, n_layer):
    """yield (phase, layer, experts_tuple) in execution order, input by input.

    Prompt: llama.cpp runs the prompt as one batch layer by layer, so within a layer the
    tokens come in position order. Generation: one token at a time through all layers.
    Per-layer caches are independent, so the cross-layer interleaving does not matter.
    """
    k = 8
    for _, inp in df.groupby("input_id", sort=True):
        for phase in ("prompt", "gen"):
            p = inp[inp["phase"] == phase]
            if p.empty:
                continue
            ex = p["expert"].to_numpy().reshape(-1, k)
            lay = p["layer"].to_numpy()[::k]
            pos = p["token_pos"].to_numpy()[::k]
            if phase == "prompt":
                order = np.lexsort((pos, lay))
            else:
                order = np.lexsort((lay, pos))
            for i in order:
                yield phase, pos[i], int(lay[i]), ex[i]


class LRU:
    def __init__(self, b):
        self.b, self.c = b, OrderedDict()

    def access(self, experts):
        n_miss = 0
        for e in experts:
            if e in self.c:
                self.c.move_to_end(e)
            else:
                n_miss += 1
                self.c[e] = True
                if len(self.c) > self.b:
                    self.c.popitem(last=False)
        return n_miss


class CRF:
    """Simplified version of llama.cpp's MoE cache policy (src/llama-moe-cache.cpp):
    combined recency-frequency score = uses decayed with a 128-token half-life, kept across
    evictions; evict the cached expert with the lowest current score. llama.cpp groups
    scores into log2 classes with an LRU list each and shares slots across layers; this
    keeps only the scoring idea, per layer."""

    HALF_LIFE = 128.0

    def __init__(self, b):
        self.b, self.cached, self.crf, self.last, self.clock = b, set(), {}, {}, 0

    def score(self, e):
        if e not in self.crf:
            return 0.0
        return self.crf[e] * 2.0 ** (-(self.clock - self.last[e]) / self.HALF_LIFE)

    def access(self, experts):
        self.clock += 1
        n_miss = 0
        for e in experts:
            self.crf[e] = self.score(e) + 1.0
            self.last[e] = self.clock
            if e not in self.cached:
                n_miss += 1
                self.cached.add(e)
        while len(self.cached) > self.b:
            # never evict an expert this token just used, unless the budget is smaller than top-k
            pool = [x for x in self.cached if x not in experts] or list(self.cached)
            self.cached.remove(min(pool, key=self.score))
        return n_miss


def make_cache(policy, b):
    return {"lru": LRU, "crf": CRF}[policy](b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ROOT / "results")
    ap.add_argument("--n-expert", type=int, default=64)
    args = ap.parse_args()
    res = Path(args.results)
    n_expert = args.n_expert

    tr = pd.read_parquet(res / "routing_trace.parquet")
    tr = tr[tr["task"].astype(str) == TASK_A].copy()
    tr["phase"] = tr["phase"].astype(str)
    n_layer = int(tr["layer"].max()) + 1
    ids = np.sort(tr["input_id"].unique())
    half = len(ids) // 2
    df_a = tr[tr["input_id"].isin(ids[:half])]
    df_b = tr[tr["input_id"].isin(ids[half:])]

    # static hot set from half A generated tokens
    gen_a = df_a[df_a["phase"] == "gen"]
    m_a = np.zeros((n_layer, n_expert), dtype=np.int64)
    np.add.at(m_a, (gen_a["layer"].to_numpy(), gen_a["expert"].to_numpy()), 1)

    gen_b = df_b[df_b["phase"] == "gen"]
    n_gen_b = len(gen_b) // 8 // n_layer

    out = {
        "note": "SIMULATION. Unit = one expert, per-layer budget. See script docstring.",
        "half_b_inputs": int(len(ids) - half),
        "half_b_gen_tokens": int(n_gen_b),
        "budgets": {},
    }

    # per-input replay so token keys are unique: run the streams input by input, keeping cache state
    def replay(policy, b, warm, touch_prompt=True):
        caches = [make_cache(policy, b) for _ in range(n_layer)]
        tok_miss = []

        def feed(df, count):
            for _, inp in df.groupby("input_id", sort=True):
                per_tok = {}
                for phase, pos, layer, experts in token_stream(inp, n_layer):
                    if phase == "prompt" and not touch_prompt:
                        continue
                    n_miss = caches[layer].access([int(e) for e in experts])
                    if count and phase == "gen":
                        per_tok[pos] = per_tok.get(pos, 0) + n_miss
                tok_miss.extend(per_tok.values())

        if warm:
            feed(df_a, count=False)
        feed(df_b, count=True)
        return np.array(tok_miss)

    for f in BUDGETS:
        b = round(f * n_expert)
        hot = np.zeros((n_layer, n_expert), dtype=bool)
        for layer in range(n_layer):
            hot[layer, np.lexsort((np.arange(n_expert), -m_a[layer]))[:b]] = True
        cold = ~hot[gen_b["layer"].to_numpy(), gen_b["expert"].to_numpy()]
        key = gen_b["input_id"].to_numpy().astype(np.int64) * 1_000_000 + gen_b["token_pos"].to_numpy()
        static = pd.Series(cold).groupby(key).sum().to_numpy()

        entry = {"experts_per_layer": b}
        for name, arr in [
            ("static_hot_set", static),
            ("lru_cold_start", replay("lru", b, warm=False)),
            ("lru_warm", replay("lru", b, warm=True)),
            ("lru_gen_only_warm", replay("lru", b, warm=True, touch_prompt=False)),
            ("crf_warm", replay("crf", b, warm=True)),
            ("crf_gen_only_warm", replay("crf", b, warm=True, touch_prompt=False)),
        ]:
            assert len(arr) == n_gen_b, (name, len(arr), n_gen_b)
            entry[name] = {
                "misses_per_gen_token_mean": round(float(arr.mean()), 2),
                "share_tokens_zero_misses_pct": round(float((arr == 0).mean() * 100), 2),
            }
        out["budgets"][f"{int(f*100)}pct"] = entry

    (res / "cache_sim.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

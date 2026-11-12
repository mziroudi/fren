"""SIMULATION: does a task profile help an adaptive expert cache?

Week 1 found that a static hot set barely beats LRU and loses to a policy like llama.cpp's
MoE cache. This asks whether the profile helps *on top of* those adaptive caches, in two ways:

  seeded : the cache starts filled with the profile's top-b experts per layer (and, for CRF,
           with scores set from profile frequencies) instead of empty. After that it
           adapts as usual. Only warm-up can improve.
  pinned : half of the budget is pinned to the profile's top experts (never evicted); the
           other half is the adaptive cache, over the remaining experts.

Two adaptive policies, as in analysis/cache_sim.py:
  lru  stand-in for the OS page cache (RAM tier). Prompt tokens touch it: the prompt reads
       the expert pages through the same mmap.
  crf  simplified llama.cpp --moe-cache-mib policy (VRAM tier). Prompt tokens do NOT touch it:
       llama.cpp skips its MoE cache for ubatches over 32 tokens
       (src/llama-moe-cache.cpp, max_batch = 32), so only generation updates it.

Two scenarios, replaying task A half B (held-out) with the profile from half A:
  fresh        cache reset before every input (a new process per request, e.g. llama-cli)
  long_running cache state carried across inputs, after first replaying half A
               (a server that has been up a while). Every policy sees the same history.

Metric: cold experts per generated token, summed over the 16 layers (128 selections per
token), mean over all generated tokens; plus the mean over the first 16 generated tokens of
each input, where warm-up matters. Same simplifications as analysis/cache_sim.py.

Writes results/seeded_cache.json and prints it.
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from cache_sim import CRF, LRU, token_stream

ROOT = Path(__file__).resolve().parent.parent
TASK_A = "code_explain"
BUDGETS = [0.10, 0.20, 0.30, 0.50]
EARLY = 16  # "first tokens" window
CRF_STEADY = 1.0 / (1.0 - 2.0 ** (-1.0 / CRF.HALF_LIFE))  # steady CRF of an expert used every token


class Pinned:
    """pinned experts always hit; the remaining slots are an adaptive cache over the rest"""

    def __init__(self, pinned, inner):
        self.pinned, self.inner = pinned, inner

    def access(self, experts):
        return self.inner.access([e for e in experts if e not in self.pinned])


def seed_lru(cache, ranked):
    # most popular last, so it is the most recently used
    cache.c = OrderedDict((e, True) for e in reversed(ranked[: cache.b]))
    return cache


def seed_crf(cache, ranked, freq):
    for e in ranked[: cache.b]:
        cache.cached.add(e)
    for e, q in enumerate(freq):
        if q > 0:
            cache.crf[e] = q * CRF_STEADY
            cache.last[e] = 0
    return cache


def build(policy, variant, b, ranked, freq):
    """one layer's cache. ranked: experts by profile count, most used first. freq: uses per token."""
    base = {"lru": LRU, "crf": CRF}[policy]
    if variant == "plain":
        return base(b)
    if variant == "seeded":
        return seed_lru(base(b), ranked) if policy == "lru" else seed_crf(base(b), ranked, freq)
    if variant == "pinned":
        n_pin = b // 2
        pinned = set(ranked[:n_pin])
        return Pinned(pinned, base(b - n_pin))
    raise ValueError(variant)


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

    # profile from half A generated tokens
    gen_a = df_a[df_a["phase"] == "gen"]
    m_a = np.zeros((n_layer, n_expert), dtype=np.int64)
    np.add.at(m_a, (gen_a["layer"].to_numpy(), gen_a["expert"].to_numpy()), 1)
    n_tok_a = len(gen_a) // 8 // n_layer
    ranked = [list(np.lexsort((np.arange(n_expert), -m_a[l]))) for l in range(n_layer)]
    freq = m_a / n_tok_a

    # pre-expand the per-input access streams once
    def streams(df):
        return [
            [(ph, int(pos), layer, [int(e) for e in ex]) for ph, pos, layer, ex in token_stream(inp, n_layer)]
            for _, inp in df.groupby("input_id", sort=True)
        ]

    s_a, s_b = streams(df_a), streams(df_b)

    def replay(policy, variant, b, scenario):
        touch_prompt = policy == "lru"
        new = lambda: [build(policy, variant, b, ranked[l], freq[l]) for l in range(n_layer)]  # noqa: E731
        caches = new()

        def feed(stream, per_tok):
            for ph, pos, layer, ex in stream:
                if ph == "prompt" and not touch_prompt:
                    continue
                n = caches[layer].access(ex)
                if per_tok is not None and ph == "gen":
                    per_tok[pos] = per_tok.get(pos, 0) + n

        if scenario == "long_running":
            for st in s_a:
                feed(st, None)
        all_tok, early = [], []
        for st in s_b:
            if scenario == "fresh":
                caches = new()
            per_tok = {}
            feed(st, per_tok)
            vals = [per_tok[p] for p in sorted(per_tok)]
            all_tok += vals
            early += vals[:EARLY]
        return np.array(all_tok), np.array(early)

    out = {
        "note": "SIMULATION. See script docstring for policies and simplifications.",
        "profile": "task A half A generated tokens", "replayed": "task A half B",
        "half_b_inputs": len(s_b), "early_window_tokens": EARLY, "budgets": {},
    }
    for f in BUDGETS:
        b = round(f * n_expert)
        entry = {"experts_per_layer": b, "pinned_per_layer": b // 2}
        # the week-1 static hot set, for reference
        hot = np.zeros((n_layer, n_expert), dtype=bool)
        for l in range(n_layer):
            hot[l, ranked[l][:b]] = True
        static = {}
        for st in s_b:
            per_tok = {}
            for ph, pos, layer, ex in st:
                if ph == "gen":
                    per_tok[pos] = per_tok.get(pos, 0) + sum(not hot[layer, e] for e in ex)
            static.setdefault("all", []).extend(per_tok[p] for p in sorted(per_tok))
            static.setdefault("early", []).extend([per_tok[p] for p in sorted(per_tok)][:EARLY])
        entry["static_hot_set"] = {
            "misses_per_gen_token": round(float(np.mean(static["all"])), 2),
            "misses_first_tokens": round(float(np.mean(static["early"])), 2),
        }
        for scenario in ("fresh", "long_running"):
            for policy in ("lru", "crf"):
                for variant in ("plain", "seeded", "pinned"):
                    a, e = replay(policy, variant, b, scenario)
                    entry[f"{scenario}/{policy}/{variant}"] = {
                        "misses_per_gen_token": round(float(a.mean()), 2),
                        "misses_first_tokens": round(float(e.mean()), 2),
                    }
        out["budgets"][f"{int(f*100)}pct"] = entry
        print(f"budget {int(f*100)}% done", flush=True)

    (res / "seeded_cache.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

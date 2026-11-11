"""Week-1 routing analysis: coverage, stability, task specificity, misses per token.

Reads results/routing_trace.parquet (from analysis/to_parquet.py).
Writes:
  results/coverage.png            coverage curve per layer
  results/coverage_by_layer.csv   experts needed for 50/80/95% of selections, per task/phase/layer
  results/coverage_summary.json   every number quoted in results/WEEK1.md

Definitions
  - "selection": one (token, layer, expert) triple. Each token makes top-k selections per layer.
  - "hot set of size b" for a layer: the b experts with the most selections in the profile data.
  - Task A inputs are split by input_id: half A = first half, half B = second half. The
    inputs are standard-library modules in a seeded random order, so the halves are not
    sorted by topic.
  - Generated tokens ("gen") are the target: that is where the speedup would be.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASK_A, TASK_B = "code_explain", "news_summary"
BUDGETS = [0.10, 0.20, 0.30, 0.50]
LEVELS = [0.50, 0.80, 0.95]


def counts_matrix(df, n_layer, n_expert):
    """selection counts, shape [n_layer, n_expert]"""
    m = np.zeros((n_layer, n_expert), dtype=np.int64)
    np.add.at(m, (df["layer"].to_numpy(), df["expert"].to_numpy()), 1)
    return m


def experts_for_level(row_counts, level):
    s = np.sort(row_counts)[::-1]
    cum = np.cumsum(s) / s.sum()
    return int(np.searchsorted(cum, level - 1e-12) + 1)


def hot_sets(m, b):
    """boolean [n_layer, n_expert], top-b experts per layer (ties broken by expert id)"""
    hot = np.zeros_like(m, dtype=bool)
    for layer in range(m.shape[0]):
        order = np.lexsort((np.arange(m.shape[1]), -m[layer]))
        hot[layer, order[:b]] = True
    return hot


def coverage_of(hot, m):
    """share of selections in m that fall in hot, per layer"""
    return (m * hot).sum(axis=1) / m.sum(axis=1)


def misses_per_token(df, hot, n_layer):
    """for each generated token: number of selected experts not in the hot set, summed over layers"""
    cold = ~hot[df["layer"].to_numpy(), df["expert"].to_numpy()]
    key = df["input_id"].to_numpy().astype(np.int64) * 1_000_000 + df["token_pos"].to_numpy()
    per_tok = pd.Series(cold).groupby(key).sum()
    return per_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ROOT / "results")
    ap.add_argument("--n-expert", type=int, default=64)
    args = ap.parse_args()
    res = Path(args.results)
    n_expert = args.n_expert

    tr = pd.read_parquet(res / "routing_trace.parquet")
    tr["task"] = tr["task"].astype(str)
    tr["phase"] = tr["phase"].astype(str)
    n_layer = int(tr["layer"].max()) + 1
    k = int(tr.groupby(["task", "input_id", "phase", "token_pos", "layer"]).size().iloc[0])
    out = {"n_layer": n_layer, "n_expert": n_expert, "top_k": k}

    # ---------- 1. coverage per layer ----------
    rows = []
    mats = {}
    for (task, phase), df in tr.groupby(["task", "phase"]):
        m = counts_matrix(df, n_layer, n_expert)
        mats[(task, phase)] = m
        for layer in range(n_layer):
            r = {"task": task, "phase": phase, "layer": layer, "selections": int(m[layer].sum())}
            for lv in LEVELS:
                r[f"pct_experts_for_{int(lv*100)}"] = 100 * experts_for_level(m[layer], lv) / n_expert
            rows.append(r)
    cov = pd.DataFrame(rows)
    cov.to_csv(res / "coverage_by_layer.csv", index=False, float_format="%.1f")
    summ = {}
    for (task, phase), g in cov.groupby(["task", "phase"]):
        if phase == "prompt":
            g = g[g["layer"] < n_layer - 1]  # last layer only sees the final prompt token
        summ[f"{task}/{phase}"] = {
            f"{stat}_pct_experts_for_{int(lv*100)}": round(float(getattr(g[f"pct_experts_for_{int(lv*100)}"], stat)()), 1)
            for lv in LEVELS
            for stat in ("median", "min", "max")
        }
    out["coverage"] = summ
    out["uniform_pct_experts_for"] = {str(int(lv * 100)): lv * 100 for lv in LEVELS}

    # ---------- 2. stability: half A -> half B, task A, generated tokens ----------
    a_gen = tr[(tr.task == TASK_A) & (tr.phase == "gen")]
    ids = np.sort(a_gen["input_id"].unique())
    half = len(ids) // 2
    ids_a, ids_b = set(ids[:half]), set(ids[half:])
    gen_a = a_gen[a_gen["input_id"].isin(ids_a)]
    gen_b = a_gen[a_gen["input_id"].isin(ids_b)]
    m_a = counts_matrix(gen_a, n_layer, n_expert)
    m_b = counts_matrix(gen_b, n_layer, n_expert)
    out["split"] = {"half_a_inputs": len(ids_a), "half_b_inputs": len(ids_b),
                    "half_a_gen_tokens": int(len(gen_a) / k / n_layer),
                    "half_b_gen_tokens": int(len(gen_b) / k / n_layer)}

    stab = {}
    for f in BUDGETS:
        b = round(f * n_expert)
        hot_a = hot_sets(m_a, b)
        stab[f"{int(f*100)}pct"] = {
            "experts_per_layer": b,
            "coverage_on_half_a_mean": round(float(coverage_of(hot_a, m_a).mean() * 100), 1),
            "coverage_on_half_b_mean": round(float(coverage_of(hot_a, m_b).mean() * 100), 1),
            "coverage_on_half_b_min_layer": round(float(coverage_of(hot_a, m_b).min() * 100), 1),
            "oracle_coverage_half_b_mean": round(float(coverage_of(hot_sets(m_b, b), m_b).mean() * 100), 1),
            "uniform_coverage": round(100 * b / n_expert, 1),
            "jaccard_hot_a_vs_hot_b_mean": round(float(np.mean([
                (hot_a[l] & hot_sets(m_b, b)[l]).sum() / (hot_a[l] | hot_sets(m_b, b)[l]).sum()
                for l in range(n_layer)])), 2),
        }
    out["stability_task_a_gen"] = stab

    # ---------- 3. task specificity: task A hot set on task B, generated tokens ----------
    m_A = mats[(TASK_A, "gen")]
    m_B = mats[(TASK_B, "gen")]
    spec = {}
    for f in BUDGETS:
        b = round(f * n_expert)
        hA, hB = hot_sets(m_A, b), hot_sets(m_B, b)
        spec[f"{int(f*100)}pct"] = {
            "experts_per_layer": b,
            "task_a_hot_on_task_a": round(float(coverage_of(hA, m_A).mean() * 100), 1),
            "task_a_hot_on_task_b": round(float(coverage_of(hA, m_B).mean() * 100), 1),
            "task_b_hot_on_task_b": round(float(coverage_of(hB, m_B).mean() * 100), 1),
            "jaccard_mean": round(float(np.mean([(hA[l] & hB[l]).sum() / (hA[l] | hB[l]).sum() for l in range(n_layer)])), 2),
        }
    out["task_specificity_gen"] = spec

    # ---------- 4. misses per generated token (hot set from half A, measured on half B) ----------
    miss = {}
    for f in BUDGETS:
        b = round(f * n_expert)
        per_tok = misses_per_token(gen_b, hot_sets(m_a, b), n_layer)
        miss[f"{int(f*100)}pct"] = {
            "experts_per_layer": b,
            "cold_experts_per_token_mean": round(float(per_tok.mean()), 2),
            "cold_experts_per_token_median": float(per_tok.median()),
            "selections_per_token": k * n_layer,
            "share_tokens_zero_cold_pct": round(float((per_tok == 0).mean() * 100), 2),
            "share_layer_visits_all_hot_pct": round(float(
                (~hot_sets(m_a, b)[gen_b["layer"].to_numpy(), gen_b["expert"].to_numpy()])
                .reshape(-1, k).any(axis=1).__invert__().mean() * 100), 1),
        }
    out["misses_static_hot_set"] = miss

    (res / "coverage_summary.json").write_text(json.dumps(out, indent=2))

    # ---------- chart ----------
    series = [
        ((TASK_A, "gen"), "Code explain, generated", "#2a78d6", "-"),
        ((TASK_A, "prompt"), "Code explain, prompt", "#2a78d6", ":"),
        ((TASK_B, "gen"), "News summary, generated", "#eb6834", "-"),
    ]
    cols = 4
    rows_n = int(np.ceil(n_layer / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(13, 3.0 * rows_n), sharex=True, sharey=True)
    x = np.arange(0, n_expert + 1) / n_expert * 100
    for layer, ax in enumerate(axes.flat):
        if layer >= n_layer:
            ax.axis("off")
            continue
        ax.plot([0, 100], [0, 100], color="#8a8985", lw=1, ls="--", label="Uniform routing")
        for key, label, color, ls in series:
            if key not in mats or (key[1] == "prompt" and layer == n_layer - 1):
                continue
            s = np.sort(mats[key][layer])[::-1]
            y = np.concatenate([[0], np.cumsum(s) / s.sum() * 100])
            ax.plot(x, y, color=color, lw=2, ls=ls, label=label)
        ax.axhline(80, color="#d9d8d3", lw=0.8, zorder=0)
        ax.set_title(f"Layer {layer}", fontsize=10, color="#0b0b0b", loc="left")
        ax.grid(color="#ecebe7", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("% of experts (most used first)", fontsize=9, color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of selections", fontsize=9, color="#52514e")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(
        f"Expert coverage per layer, OLMoE-1B-7B-0125-Instruct Q8_0 ({n_expert} experts, top-{k})",
        y=0.995, fontsize=12, color="#0b0b0b", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(res / "coverage.png", dpi=110, facecolor="#fcfcfb")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

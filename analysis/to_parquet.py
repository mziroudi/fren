"""Convert raw route_trace TSVs into parquet and run sanity checks.

Writes:
  results/routing_trace.parquet  task, input_id, token_pos, phase, layer, expert
  results/routing.parquet        task, phase, layer, expert, count

Sanity checks (the script fails if any is broken):
  - every (token, layer) row has exactly top-k distinct experts in [0, n_expert)
  - every generated token has all layers
  - every prompt token has all layers except the last one, which llama.cpp only computes
    for the final prompt token (its output is the only one needed for sampling)
  - counts in routing.parquet sum to rows * top-k
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(raw_dir):
    frames = []
    for p in sorted(Path(raw_dir).glob("trace_*.tsv")):
        df = pd.read_csv(p, sep="\t", dtype={"task": "category", "phase": "category"})
        frames.append(df)
    if not frames:
        sys.exit(f"no trace_*.tsv files in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=ROOT / "results/raw")
    ap.add_argument("--out", default=ROOT / "results")
    ap.add_argument("--n-expert", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out)

    rows = load(args.raw)
    experts = np.array([list(map(int, s.split(","))) for s in rows["experts"]], dtype=np.int16)
    k = experts.shape[1]
    n_layer = int(rows["layer"].max()) + 1

    # check: k distinct experts in range, per row
    assert experts.min() >= 0 and experts.max() < args.n_expert, "expert id out of range"
    srt = np.sort(experts, axis=1)
    assert (np.diff(srt, axis=1) > 0).all(), "duplicate expert in a row"

    # check: layers per token
    per_tok = rows.groupby(["task", "input_id", "phase", "token_pos"], observed=True)["layer"].nunique()
    gen = per_tok.xs("gen", level="phase")
    assert (gen == n_layer).all(), "a generated token is missing layers"
    prm = per_tok.xs("prompt", level="phase")
    last_pos = prm.groupby(level=["task", "input_id"], observed=True).transform(
        lambda s: s.index.get_level_values("token_pos").max()
    )
    is_last = prm.index.get_level_values("token_pos") == last_pos.values
    assert (prm[~is_last] == n_layer - 1).all(), "prompt token layer count unexpected"
    assert (prm[is_last] == n_layer).all(), "final prompt token should have all layers"

    trace = pd.DataFrame(
        {
            "task": np.repeat(rows["task"].to_numpy(), k),
            "input_id": np.repeat(rows["input_id"].to_numpy(np.int16), k),
            "token_pos": np.repeat(rows["token_pos"].to_numpy(np.int32), k),
            "phase": np.repeat(rows["phase"].to_numpy(), k),
            "layer": np.repeat(rows["layer"].to_numpy(np.int8), k),
            "expert": experts.reshape(-1),
        }
    )
    trace["task"] = trace["task"].astype("category")
    trace["phase"] = trace["phase"].astype("category")
    trace.to_parquet(out / "routing_trace.parquet", index=False)

    counts = (
        trace.groupby(["task", "phase", "layer", "expert"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    assert counts["count"].sum() == len(rows) * k
    counts.to_parquet(out / "routing.parquet", index=False)

    n_tok = per_tok.groupby(level=["task", "phase"], observed=True).size()
    n_inp = rows.groupby("task", observed=True)["input_id"].nunique()
    print(f"top-k={k}, layers={n_layer}, trace rows={len(trace):,}")
    print("inputs per task:\n" + n_inp.to_string())
    print("tokens per task/phase:\n" + n_tok.to_string())
    print("all sanity checks passed")


if __name__ == "__main__":
    main()

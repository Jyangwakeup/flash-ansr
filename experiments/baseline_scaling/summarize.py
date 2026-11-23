#!/usr/bin/env python
"""Summarize runs/baseline_scaling/seed0/baseline.jsonl -> X%, the solve-rate-vs-N curve, saturation tests.

Writes summary.json, solve_rate_vs_N.png and unique_vs_N.png next to the input file.

    python experiments/baseline_scaling/summarize.py [--in runs/baseline_scaling/seed0/baseline.jsonl]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from srbf.metrics.paired import mcnemar_exact, wilson_interval

ROOT = Path(__file__).resolve().parents[2]
X_PERCENT_N = 1000
EARLY_MAX_N = 100        # "solved early": already at N <= 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "runs" / "baseline_scaling" / "seed0" / "baseline.jsonl"))
    args = ap.parse_args()
    inp = Path(args.inp)
    rows = [json.loads(line) for line in inp.read_text().splitlines() if line.strip()]

    problems = sorted({r["problem"] for r in rows})
    grid = sorted({r["draws"] for r in rows})
    by = defaultdict(list)                       # (N, seed) -> rows
    for r in rows:
        by[(r["draws"], r["seed"])].append(r)

    def solved_vec(n: int, seed: int, key: str) -> np.ndarray | None:
        got = {r["problem"]: bool(r.get(key)) for r in by.get((n, seed), [])}
        return np.array([got[p] for p in problems]) if len(got) == len(problems) else None

    summary: dict = {"input": str(inp), "n_problems": len(problems), "per_N": {}, "adjacent_mcnemar": {}}
    for n in grid:
        seeds = sorted({s for (nn, s) in by if nn == n})
        entry: dict = {"seeds": seeds}
        for key in ("solved_oracle", "solved_top1"):
            per_seed = [v.mean() for s in seeds if (v := solved_vec(n, s, key)) is not None]
            pooled = [bool(r.get(key)) for s in seeds for r in by[(n, s)]]
            lo, hi = wilson_interval(sum(pooled), len(pooled))
            entry[key] = {"rate": float(np.mean(pooled)) if pooled else float("nan"),
                          "seed_std": float(np.std(per_seed)) if len(per_seed) > 1 else None,
                          "wilson95": [lo, hi], "n_rows": len(pooled)}
        ok = [r for s in seeds for r in by[(n, s)] if "error" not in r]
        entry["mean_unique"] = float(np.mean([r["n_unique"] for r in ok])) if ok else None
        entry["mean_flops"] = float(np.mean([r["flops"] for r in ok])) if ok else None
        entry["mean_fit_time_s"] = float(np.mean([r["fit_time"] for r in ok])) if ok else None
        entry["n_errors"] = sum("error" in r for s in seeds for r in by[(n, s)])
        summary["per_N"][n] = entry

    # saturation: does each step up in N still solve new problems? (paired, seed 0)
    for a, b in zip(grid, grid[1:]):
        va, vb = solved_vec(a, 0, "solved_oracle"), solved_vec(b, 0, "solved_oracle")
        if va is not None and vb is not None:
            t = mcnemar_exact(vb, va)
            summary["adjacent_mcnemar"][f"{a}->{b}"] = {
                "gained": t.n_a_only, "lost": t.n_b_only, "p_value": t.p_value}

    # per-problem: the first N at which seed 0 solves it (oracle)
    first: dict[int, int | None] = {}
    for p in problems:
        first[p] = next((n for n in grid
                         if any(r["problem"] == p and r.get("solved_oracle") for r in by.get((n, 0), []))), None)
    summary["problem_classes"] = {
        f"early (N<={EARLY_MAX_N})": sum(1 for v in first.values() if v is not None and v <= EARLY_MAX_N),
        f"late (N>{EARLY_MAX_N})": sum(1 for v in first.values() if v is not None and v > EARLY_MAX_N),
        f"never (up to N={grid[-1]})": sum(1 for v in first.values() if v is None),
    }
    summary["first_solving_N"] = first
    if X_PERCENT_N in summary["per_N"]:
        summary["X_percent"] = 100 * summary["per_N"][X_PERCENT_N]["solved_oracle"]["rate"]
        summary["X_percent_top1"] = 100 * summary["per_N"][X_PERCENT_N]["solved_top1"]["rate"]

    (inp.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "first_solving_N"}, indent=2))
    plot(summary, grid, inp.parent)


def plot(summary: dict, grid: list[int], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, xkey, xlabel in ((axes[0], "N", "draws N"), (axes[1], "mean_flops", "decoder+encoder FLOPs / problem")):
        for key, label, color in (("solved_oracle", "oracle (any candidate)", "#2a6fdb"),
                                  ("solved_top1", "top-1 (returned answer)", "#e07b39")):
            xs = [n if xkey == "N" else summary["per_N"][n]["mean_flops"] for n in grid]
            ys = [100 * summary["per_N"][n][key]["rate"] for n in grid]
            lo = [100 * summary["per_N"][n][key]["wilson95"][0] for n in grid]
            hi = [100 * summary["per_N"][n][key]["wilson95"][1] for n in grid]
            ax.plot(xs, ys, "o-", color=color, label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("solved on fastsrb (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Flash-ANSR v25.0-T8-3M, independent sampling (95% Wilson)")
    fig.tight_layout()
    fig.savefig(out_dir / "solve_rate_vs_N.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(grid, [summary["per_N"][n]["mean_unique"] for n in grid], "o-", label="unique draws")
    ax.plot(grid, grid, "--", color="gray", label="y = N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("draws N")
    ax.set_ylabel("mean unique draws / problem")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "unique_vs_N.png", dpi=150)


if __name__ == "__main__":
    main()

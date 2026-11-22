#!/usr/bin/env python
"""Independent-sampling baseline: solve rate of Flash-ANSR v25.0-T8-3M vs. the draw budget N.

One JSONL row per (problem, N, seed). Resumable: rows already in --out are skipped.

Solved (both recorded, see README.md):
  * oracle -- ANY fitted candidate reproduces the target on the held-out validation points
              (FVU <= float32 eps, float64 variance; srbf.metrics.is_perfect_fit)
  * top1   -- the returned (rank-0) candidate does

Usage
-----
    python experiments/baseline_scaling/run_baseline.py --limit 5 --N 50          # smoke test
    python experiments/baseline_scaling/run_baseline.py                           # full grid
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from srbf.data_sources import CatalogSource
from srbf.metrics.numeric import fvu as srbf_fvu

from flash_ansr import FlashANSR, SoftmaxSamplingConfig, get_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from exp_wandb import Run  # noqa: E402
MODEL = "psaegert/flash-ansr-v25.0-T8-3M"
DATA_SEED = 20260825                         # the fastsrb realization used by experiments/v24_capability_matrix
DEFAULT_GRID = {50: 1, 100: 1, 300: 1, 1000: 1, 3000: 1}   # N -> number of fit seeds (1, as srbf runs do)


# ----------------------------------------------------------------------------- compute accounting
def decoder_flops_per_token(model: Any, context_len: float) -> float:
    """Forward FLOPs of ONE decoded token: 2*P_dec (matmuls) + attention scores/values.

    context_len = tokens already in the decoder sequence (prefix + generated so far, averaged).
    Self-attention: 4*d*context per layer; cross-attention: 4*d*S_mem per layer (S_mem = encoder seeds).
    """
    core = model.flash_ansr_model
    p_dec = sum(p.numel() for p in core.decoder.parameters())
    layers = len(core.decoder.layers)
    d = int(core.decoder.layers[0].self_attn.embed_dim) if hasattr(core.decoder.layers[0], "self_attn") \
        and hasattr(core.decoder.layers[0].self_attn, "embed_dim") else 192   # v25.0-T8-3M: 192
    s_mem = int(core.null_memory.shape[1]) if hasattr(core, "null_memory") else 128  # encoder seeds
    return 2.0 * p_dec + layers * 4.0 * d * (context_len + s_mem)


def encoder_flops(model: Any, n_points: int) -> float:
    """One encoder forward over the support set: ~2*P_enc FLOPs per point (ISAB keeps it linear in n)."""
    core = model.flash_ansr_model
    p_enc = sum(p.numel() for p in core.encoder.parameters())
    return 2.0 * p_enc * n_points


# ----------------------------------------------------------------------------- one problem
def evaluate(model: Any, sample: Any, draws: int, seed: int) -> dict:
    y_fit = sample.y_support
    x_val, y_val = sample.x_validation, np.asarray(sample.y_validation, dtype=np.float64).reshape(-1)
    names = sample.metadata.get("variable_names")
    row: dict = {"draws": draws, "seed": seed}

    t0 = time.time()
    with np.errstate(all="ignore"):
        result = model.fit(sample.x_support, y_fit,
                           variable_names=names if names is not None else "auto",
                           draws=draws, seed=seed)
    row["fit_time"] = time.time() - t0
    row["generation_time"] = result.generation_time
    row["refinement_time"] = result.refinement_time

    # unique beams = what generation produced after dedup (the ledger holds every unique draw)
    # ledger rows with parent != -1 are refinement-made re-spellings, not draws
    led = result.ledger
    draws_rows = {tuple(t) for t, par in zip(led.token_lists, led.parent or [-1] * len(led.token_lists))
                  if par == -1}
    lengths = [sum(1 for tok in t if tok != 0) for t in draws_rows]   # 0 = <pad>; includes the 3-token prompt
    row["n_unique"] = len(lengths)
    row["n_fitted"] = len(result.candidates)
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    row["mean_gen_len"] = mean_len
    # duplicates are not kept, so total generated tokens are estimated as draws * mean unique length
    row["gen_tokens_est"] = draws * mean_len
    row["prefix_tokens"] = 0                       # independent sampling: no conditioning prefix
    row["flops"] = (draws * mean_len * decoder_flops_per_token(model, context_len=mean_len / 2)
                    + encoder_flops(model, sample.n_support))

    # validation FVU of every fitted candidate (score-sorted, rank 0 first)
    fvus = []
    with np.errstate(all="ignore"):
        for rank in range(len(result.candidates)):
            try:
                y_pred = np.asarray(result.predict(x_val, rank=rank), dtype=np.float64).reshape(-1)
                fvus.append(float(srbf_fvu(y_val, y_pred)))
            except Exception:  # an unevaluable candidate never counts as a fit
                fvus.append(float("inf"))
    fvus_arr = np.asarray(fvus) if fvus else np.asarray([np.inf])
    eps = float(np.finfo(np.float32).eps)
    row["top1_fvu_val"] = float(fvus_arr[0])
    row["min_fvu_val"] = float(np.nanmin(np.where(np.isnan(fvus_arr), np.inf, fvus_arr)))
    row["solved_top1"] = bool(row["top1_fvu_val"] <= eps)
    row["solved_oracle"] = bool(row["min_fvu_val"] <= eps)
    hits = np.flatnonzero(fvus_arr <= eps)
    row["first_solving_rank"] = int(hits[0]) if hits.size else -1
    row["top1_expression"] = result.candidates[0].expression_infix if result.candidates else None
    return row


# ----------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "baseline_scaling" / "seed0" / "baseline.jsonl"))
    ap.add_argument("--N", type=int, nargs="*", help="draw budgets (default: 50 100 300 1000 3000)")
    seed_args = ap.add_mutually_exclusive_group()
    seed_args.add_argument("--seeds", type=int, help="number of fit seeds per N, starting at 0 (default: 1)")
    seed_args.add_argument("--seed", type=int, help="run only this fit seed")
    ap.add_argument("--limit", type=int, help="only the first LIMIT problems")
    ap.add_argument("--device", default="cpu")   # MPS has no float64, which the 64-bit IEEE754 input encoding needs
    args = ap.parse_args()

    grid = {n: (args.seeds or DEFAULT_GRID.get(n, 3)) for n in (args.N or DEFAULT_GRID)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    solved: defaultdict[int, dict[str, list]] = defaultdict(lambda: {"oracle": [], "top1": []})   # N -> per-row outcomes (incl. earlier rows)
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["problem"], r["draws"], r["seed"]))
            solved[r["draws"]]["oracle"].append(r["solved_oracle"])
            solved[r["draws"]]["top1"].append(r["solved_top1"])
    run = Run("baseline_scaling", name=out.parent.name, job_type="eval", out=out.parent,
              config=dict(model=MODEL, data_seed=DATA_SEED, grid=grid, limit=args.limit, device=args.device))

    model = FlashANSR.load(
        directory=get_path("models", MODEL),
        generation_config=SoftmaxSamplingConfig(draws=1000),
        ranking="mdl",
        compute={"device": args.device, "workers": None},
    )
    source = CatalogSource.from_catalog(
        "fastsrb", sampling={"n_support": 512, "n_validation": 512, "noise": 0.0, "seed": DATA_SEED},
        target_size=args.limit)
    source.prepare(adapter=type("A", (), {"get_simplipy_engine": lambda self: model.simplipy_engine})())

    with out.open("a") as fh:
        for idx, sample in enumerate(source):
            if sample.is_placeholder:
                print(f"[{idx}] placeholder, skipped: {sample.placeholder_reason}")
                continue
            for n, n_seeds in grid.items():
                for seed in ([args.seed] if args.seed is not None else range(n_seeds)):
                    if (idx, n, seed) in done:
                        continue
                    try:
                        row = evaluate(model, sample, n, seed)
                    except Exception as exc:   # record the failure; it counts as unsolved
                        row = {"draws": n, "seed": seed, "error": repr(exc),
                               "solved_oracle": False, "solved_top1": False}
                    row.update(problem=idx, catalog="fastsrb", model=MODEL, data_seed=DATA_SEED,
                               n_variables=int(sample.x_support.shape[1]),
                               expression=" ".join(map(str, sample.metadata.get("expression") or [])))
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    solved[n]["oracle"].append(row["solved_oracle"])
                    solved[n]["top1"].append(row["solved_top1"])
                    values = {f"N{n}/solved_oracle": float(np.mean(solved[n]["oracle"])),
                              f"N{n}/solved_top1": float(np.mean(solved[n]["top1"])),
                              f"N{n}/rows": len(solved[n]["oracle"]), "problem": idx}
                    if "fit_time" in row:
                        values[f"N{n}/fit_time"] = row["fit_time"]
                        values[f"N{n}/n_unique"] = row["n_unique"]
                    run.log(values)
                    print(f"[{idx}] N={n} s={seed} oracle={row['solved_oracle']} top1={row['solved_top1']} "
                          f"uniq={row.get('n_unique')} t={row.get('fit_time', float('nan')):.1f}s")
    model.close()
    run.table("solve_rate_vs_N", ["N", "rows", "solved_oracle", "solved_top1"],
              [[n, len(v["oracle"]), float(np.mean(v["oracle"])), float(np.mean(v["top1"]))]
               for n, v in sorted(solved.items())])
    run.summary({f"N{n}/solved_oracle": float(np.mean(v["oracle"])) for n, v in solved.items()}
                | {f"N{n}/solved_top1": float(np.mean(v["top1"])) for n, v in solved.items()})
    run.finish()


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()

"""Does a trained model use its history input? Two diagnostics on existing checkpoints.

a) Target loss with vs without history (the same sample, same rng, history ids only): if the
   loss does not move, the model ignores the history.
b) Near-correct histories built from the true formula -- exact, constants x1.3, one variable
   swapped -- versus none: loss on the target, and generation (probe draws, held-out success).
   If even the exact answer does not help, the model has not learned to read history at all.
   ``t8_candidate``: the history the evaluation would give (one of the first history_top of the
   stored T8 model pool, chosen by the eval seed), for comparison with the made-up ones.

    python experiments/history_benefits/core/diagnose.py [--groups B C] [--n 100] [--draws 50]
    python experiments/history_benefits/core/diagnose.py --config experiments/history_benefits/r3_near_correct_history/config.yaml \
        --groups C --selected        # round 3: its pre-registered selected checkpoint

Results: <output>/diagnostics/history_use.json (and printed). Not part of the locked rules: a
diagnostic added after the round-2 results.
"""

from typing import Any
import argparse
import json
from dataclasses import replace

import numpy as np
import torch
import yaml
from flash_ansr.inference import evaluate_prefix
from flash_ansr.data.streaming import build_training_sample

from common import ROOT, collate, dump, history_ids, loss, problem_ids, read_problem, source, worker_config
from run_support import load_checkpoint, load_step0
from evaluation import EPS32, choose_history, generate, outcome, wrapper_for
from common import read_pool


def variants(problem: Any) -> dict:
    """Near-correct history expressions (realized prefix) from the true formula."""
    expr = list(problem.expression)
    out = {"exact": expr}
    scaled, changed = [], False
    for t in expr:
        try:
            v = float(t)
        except ValueError:
            scaled.append(t)
            continue
        scaled.append(repr(v * 1.3))
        changed = True
    if changed:
        out["consts_x1.3"] = scaled
    used = [t for t in expr if t.startswith("x") and t[1:].isdigit()]
    others = [v for v in problem.variables if v not in used]
    if used:
        last = max(i for i, t in enumerate(expr) if t == used[-1])
        swap = list(expr)
        swap[last] = others[0] if others else (used[0] if used[0] != used[-1] else None)
        if swap[last] is not None:
            out["swap_variable"] = swap
    return out


def sample(problem: Any, wc: Any, catalog: Any, hist: Any, pred: Any, seed: Any, model: Any) -> Any:
    ids = history_ids(model, {"expression_prefix": hist}) if hist is not None else []
    s = build_training_sample(problem, replace(wc, unconditional_prob=0), catalog,
                              np.random.default_rng(seed), noise_spec=True, strict_length=True,
                              history_ids=ids)
    if s is None or "rejected" in s:
        return None
    valid = s["metadata"]["history_valid"]
    s["history"] = pred[s["row_indices"]] if valid else np.zeros(len(s["row_indices"]))
    return s


def target_ce(model: Any, s: Any) -> tuple[float, list]:
    batch = collate([s], model.tokenizer)
    labels = batch["labels"][0]
    kept = labels[labels != model.tokenizer["<pad>"]].tolist()
    with torch.no_grad():
        _, metrics = loss(model, batch)
    return float(metrics["ce"]), kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="experiments/history_benefits/r2_random_vs_model_history/config.yaml")
    parser.add_argument("--groups", nargs="+", default=["B", "C"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--selected", action="store_true",
                        help="use train/<group>/selected.json (round 3) instead of the cooldown step")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    torch.set_num_threads(cfg["threads"])
    out = ROOT / cfg["output"]
    ids = problem_ids(cfg, "validation", cfg["validation_problems"])[: args.n]
    m = load_step0(cfg, out)
    src, recipe = source(cfg)
    wc = worker_config(cfg, m, recipe)
    engine = m.simplipy_engine
    report: dict = {"problems": len(ids), "draws": args.draws}
    from wb import Run

    run = Run(cfg, out, "diagnose-" + "".join(args.groups), "diagnose",
              dict(groups=args.groups, problems=len(ids), draws=args.draws, selected=args.selected))
    for g in args.groups:
        if args.selected:
            ckpt = ROOT / json.loads((out / "train" / g / "selected.json").read_text())["checkpoint"]
        else:
            ckpt = out / "cooldown" / g / f"checkpoint_{cfg['steps']}.pt"
        load_checkpoint(m, ckpt)
        report.setdefault("checkpoints", {})[g] = str(ckpt)
        m.eval()
        wrapper = wrapper_for(cfg, m)
        losses: dict = {}
        gen: dict = {}
        try:
            for pid in ids:
                p = read_problem(out, pid)
                hists = {"none": None, **variants(p)}
                pool, _ = read_pool(out, pid, "model")
                chosen = choose_history(cfg, pid, pool)
                if chosen is not None:
                    hists["t8_candidate"] = list(pool[chosen]["expression_prefix"])
                base = sample(p, wc, src.catalog, None, None, cfg["eval_seed"] + pid, m)
                if base is None:
                    continue
                ce0, labels0 = target_ce(m, base)
                for name, expr in hists.items():
                    pred = None
                    if expr is not None:
                        try:
                            pred = np.asarray(evaluate_prefix(engine, expr, p.x_support,
                                                              n_variables=p.x_support.shape[1])).reshape(-1)
                        except Exception:  # noqa: BLE001
                            continue
                        if not np.isfinite(pred).all():
                            continue
                        s = sample(p, wc, src.catalog, expr, pred, cfg["eval_seed"] + pid, m)
                        if s is None:
                            continue
                        ce, labels = target_ce(m, s)
                        if labels != labels0:  # the target must be identical to compare losses
                            continue
                        losses.setdefault(name, []).append(ce - ce0)
                    hist = None if expr is None else ({"expression_prefix": expr}, pred)
                    r = generate(cfg, m, wrapper, p, pid, hist, args.draws)
                    o = outcome(r["candidates"])
                    gen.setdefault(name, []).append(dict(
                        pid=pid, oracle_val=o["oracle_val"], top1_val=o["top1_val"],
                        copies=sum(c["fvu_val"] <= EPS32 for c in r["candidates"]),
                        reason=r["reason"]))
                print(f"{g} {pid} done", flush=True)
                run.log({f"{g}/running_oracle/{k}": float(np.mean([x["oracle_val"] for x in v]))
                         for k, v in gen.items()}, step=len(gen.get("none", [])))
        finally:
            wrapper.close()
        report[g] = {
            "loss_delta_vs_none": {k: dict(n=len(v), mean=float(np.mean(v)),
                                           frac_lower=float(np.mean(np.array(v) < 0)))
                                   for k, v in losses.items()},
            "generation": {k: dict(n=len(v), oracle_val=float(np.mean([x["oracle_val"] for x in v])),
                                   top1_val=float(np.mean([x["top1_val"] for x in v])))
                           for k, v in gen.items()},
            "rows": gen,
        }
        print(json.dumps({k: report[g][k] for k in ("loss_delta_vs_none", "generation")}, indent=1), flush=True)
    dump(out / "diagnostics" / "history_use.json", report)
    run.summary({f"{g}/{part}/{k}": v[metric] for g in args.groups
                 for part, metric in (("generation", "oracle_val"), ("loss_delta_vs_none", "mean"))
                 for k, v in report[g][part].items()})
    run.finish()


if __name__ == "__main__":
    main()

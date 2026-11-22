"""History ablation on the validation set (exploratory; 正式实验规则.md §12.7).

The history has two parts: the formula (tokens in the sequence) and its predictions on the
support points (data column 18). For each history sample of the validation set (ids 200000..)
the same sample is scored under five conditions that differ only in the history:

    none         no history (same problem, same rows, same target)
    full         formula + its predictions
    tokens_only  formula, predictions column zeroed
    mismatched   formula, predictions of another problem's history
    perfect      formula, predictions column = the y the model sees (looks like a perfect fit)

History samples: every validation problem once per near-correct level (exact, constants,
structure_1, structure_2_3) and once with a real T8 candidate (grouped by its support FVU).
Target cross-entropy only (forward passes, no generation).

    python experiments/history_benefits/ablate.py --checkpoint runs/history_benefits/mix/train/C/checkpoint_1500.pt
    python experiments/history_benefits/ablate.py --selected      # train/C/selected.json

Results: <output>/ablation/<checkpoint name>.json (+ W&B).
"""

from typing import Any
import argparse
import json

import numpy as np
import torch
import yaml

from common import ROOT, collate, dump, eval_seed, loss, problem_ids, source, worker_config
from corrupt import LEVELS
from medium import load_checkpoint, load_step0
from mix_samples import MixSampler

CONDITIONS = ("none", "full", "tokens_only", "mismatched", "perfect")
FVU_BINS = ((0.0, 1e-3, "real fvu<1e-3"), (1e-3, 0.1, "real fvu 1e-3–0.1"), (0.1, float("inf"), "real fvu≥0.1"))


def fit_column(column: np.ndarray, n: int) -> np.ndarray:
    """``column`` resized to ``n`` rows (tiled / cut), for the mismatched condition."""
    return np.resize(np.asarray(column, dtype=float), n)


def variants(base: Any, full: Any, other: Any) -> dict | None:
    """The five conditions of one history sample, or None if the history did not fit in the
    sequence or the targets differ (they must be identical to compare losses)."""
    if not full["metadata"]["history_valid"]:
        return None
    out = {"none": base, "full": full}
    for name in ("tokens_only", "mismatched", "perfect"):
        s = dict(full)
        n = len(full["row_indices"])
        if name == "tokens_only":
            s["history"] = np.zeros(n)
        elif name == "mismatched":
            s["history"] = fit_column(other["history"], n)
        else:
            s["history"] = np.asarray(full["y_encoder"], dtype=float).reshape(-1)
        out[name] = s
    return out


def target_ces(model: Any, samples: list, micro: int) -> list:
    ces: list = []
    for begin in range(0, len(samples), micro):
        _, metrics = loss(model, collate(samples[begin: begin + micro], model.tokenizer))
        ces += metrics["ce_rows"]
    return ces


def labels_of(model: Any, s: Any) -> list:
    labels = collate([s], model.tokenizer)["labels"][0]
    return labels[labels != model.tokenizer["<pad>"]].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="experimental/history_benefits/mix.yaml")
    parser.add_argument("--checkpoint", help="path relative to the repo root")
    parser.add_argument("--selected", action="store_true", help="the pre-registered selected C checkpoint")
    parser.add_argument("--group", default="C")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    torch.set_num_threads(args.threads)
    out = ROOT / cfg["output"]
    if args.selected:
        ckpt = ROOT / json.loads((out / "train" / args.group / "selected.json").read_text())["checkpoint"]
    else:
        ckpt = ROOT / args.checkpoint
    m = load_step0(cfg, out)
    load_checkpoint(m, ckpt)
    m.eval()
    src, recipe = source(cfg)
    wc = worker_config(cfg, m, recipe)
    sampler = MixSampler(cfg, out, m, wc, src.catalog, None, args.group, problem_ids(cfg, "train", cfg["train_problems"]))
    ids = problem_ids(cfg, "selection", cfg["selection_problems"])

    # build: (group label, five-condition dict)
    built: list = []
    skipped = 0
    for kind, level in [("near", lv) for lv in LEVELS] + [("real", None)]:
        pairs = []
        for pid in ids:
            seed = [eval_seed(cfg, pid), 7, LEVELS.index(level) if level else 9]
            full = sampler.make(kind, np.random.default_rng(seed), validation=True, pid=pid, level=level)
            base = sampler.make(kind, np.random.default_rng(seed), validation=True, pid=pid, level=level,
                                drop_history=True)
            if full["info"].get("pid") != pid or labels_of(m, full) != labels_of(m, base):
                skipped += 1  # an empty pool fell back to another problem, or targets differ
                continue
            pairs.append((full, base))
        for i, (full, base) in enumerate(pairs):
            other = pairs[(i + 1) % len(pairs)][0]  # the next problem's history predictions
            v = variants(base, full, other)
            if v is None:
                skipped += 1
                continue
            if kind == "near":
                label = f"near {full['info']['actual']}"
            else:
                f = full["info"]["history_fvu"]
                label = next(name for lo, hi, name in FVU_BINS if lo <= (f if f is not None else np.inf) < hi)
            built.append((label, v))

    # score: one forward pass per condition per sample
    ces = {c: target_ces(m, [v[c] for _, v in built], cfg["micro_batch"]) for c in CONDITIONS}
    groups: dict = {}
    for i, (label, _) in enumerate(built):
        groups.setdefault(label, []).append(i)
    table = {}
    for label, idx in sorted(groups.items()):
        row = {c: float(np.mean([ces[c][i] for i in idx])) for c in CONDITIONS}
        row["n"] = len(idx)
        row["helped_vs_none"] = float(np.mean([ces["full"][i] < ces["none"][i] for i in idx]))
        table[label] = row
    result = dict(checkpoint=str(ckpt.relative_to(ROOT)), problems=len(ids), skipped=skipped,
                  conditions=list(CONDITIONS), table=table)
    dump(out / "ablation" / f"{ckpt.parent.name}_{ckpt.stem}.json", result)
    print(f"{'group':22s} {'n':>4s} " + " ".join(f"{c:>11s}" for c in CONDITIONS) + "  full<none")
    for label, row in table.items():
        print(f"{label:22s} {row['n']:4d} " + " ".join(f"{row[c]:11.3f}" for c in CONDITIONS)
              + f"  {100 * row['helped_vs_none']:5.1f}%")

    from wb import Run

    run = Run(cfg, out, f"ablation-{ckpt.parent.name}-{ckpt.stem}", "ablation", dict(checkpoint=result["checkpoint"]))
    run.table("ablation/target_ce", ["group", "n", *CONDITIONS, "full<none"],
              [[k, r["n"], *(r[c] for c in CONDITIONS), r["helped_vs_none"]] for k, r in table.items()])
    run.summary({f"{k}/{c}": r[c] for k, r in table.items() for c in CONDITIONS})
    run.finish()


if __name__ == "__main__":
    main()

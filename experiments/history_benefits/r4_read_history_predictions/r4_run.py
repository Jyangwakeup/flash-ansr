"""Round 4: can the model learn to read the history predictions? (设计与预注册.md §14)

Two groups trained 1,000 steps from a step-0 whose history-column input weights are re-drawn
(no longer zero): I (round-3 C's sample mix), IC (near-correct histories balanced over FVU
bins). Judged by the history input ablation on the validation set (ids 200000..) at step 1,000,
against round-3 C's step 1,000 (same steps, same learning-rate history). Every stage resumes.

    link                    problems + model pools from round 3 (hard links); step-0 = round 3's
                            with the history column re-initialized; checks -> link_complete.json
    train --group I|IC      1,000 steps, checkpoint + evaluation-set loss every 250
    ablate --which NAME     input ablation of one checkpoint (r3_C, I, IC) -> ablation/NAME.json
    copy-probe --which NAME teacher-forced copy check with the exact formula -> copy_probe/NAME.json
    report                  tables -> runs/.../报告.md (+ W&B)
"""

from pathlib import Path
from typing import Any
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))  # shared code: experiments/history_benefits/core
from common import ROOT, dump, eval_seed, problem_ids, sha, source, worker_config  # noqa: E402
from run_support import load_checkpoint, load_step0, provenance  # noqa: E402

CONFIG = "experiments/history_benefits/r4_read_history_predictions/config.yaml"
GROUPS = ("I", "IC")
WHICH = ("r3_C", "I", "IC")
SAME_AS_REUSE = ("seed", "data_seed_train", "data_seed_validation", "data_seed_selection", "data_seed_fresh",
                 "train_seed", "fit_seed", "eval_seed", "recipe", "model", "draws", "restarts", "keep",
                 "validation_sampling", "history_top", "max_length")


def ids(cfg: Any, split: str) -> list[int]:
    count = {"train": "train_problems", "validation": "validation_problems",
             "selection": "selection_problems"}[split]
    return problem_ids(cfg, split, cfg[count])


def link_file(src: Any, dst: Any) -> None:
    if dst.exists():
        return
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def sampler_for(cfg: Any, out: Any, m: Any, group: str, fresh: Any = None) -> Any:
    from r4_samples import MixSampler

    src, recipe = source(cfg)
    return MixSampler(cfg, out, m, worker_config(cfg, m, recipe), src.catalog, fresh, group, ids(cfg, "train"))


def link(cfg: Any, out: Any) -> None:
    """Problems and model pools of round 3; step-0 with the history column re-initialized."""
    from r4_samples import column_norms, eval_set, target_ces

    reuse = ROOT / cfg["reuse"]
    old = yaml.safe_load((reuse / "config.yaml").read_text())
    for key in SAME_AS_REUSE:
        if old.get(key) != cfg.get(key):
            raise ValueError(f"{key} differs from the reused run: {old.get(key)} vs {cfg.get(key)}")
    wanted = ids(cfg, "train") + ids(cfg, "validation") + ids(cfg, "selection")
    for pid in wanted:
        for suffix in (".json", ".npz"):
            link_file(reuse / "problems" / f"{pid}{suffix}", out / "problems" / f"{pid}{suffix}")
            link_file(reuse / "pools" / "model" / f"{pid}{suffix}", out / "pools" / "model" / f"{pid}{suffix}")
    shutil.copy2(reuse / "model_config.json", out / "model_config.json")

    # step-0: round 3's, history column (the last data column of the encoder input) re-drawn
    state = torch.load(reuse / "step0.pt", weights_only=True)
    old_model = load_step0(cfg, reuse)
    e = old_model.pre_encoder.encoding_size
    w = state["encoder.embedding.weight"]
    n_cols = w.shape[1] // e
    std = float(w[:, : (n_cols - 2) * e].std())  # the x columns
    gen = torch.Generator().manual_seed(int(cfg["history_init_seed"]))
    new_state = {k: v.clone() for k, v in state.items()}
    new_state["encoder.embedding.weight"][:, (n_cols - 1) * e:] = (
        torch.randn(w.shape[0], e, generator=gen, dtype=torch.float64) * std).to(w.dtype)
    torch.save(new_state, out / "step0.pt")
    others = all(torch.equal(state[k], new_state[k]) for k in state if k != "encoder.embedding.weight")
    others = others and torch.equal(state["encoder.embedding.weight"][:, : (n_cols - 1) * e],
                                    new_state["encoder.embedding.weight"][:, : (n_cols - 1) * e])
    new_model = load_step0(cfg, out)

    # step-0 losses: no-history samples must be unchanged (the history column is masked)
    old_model.eval()
    new_model.eval()
    sampler = sampler_for(cfg, out, new_model, "IC")
    rows = eval_set(sampler, ids(cfg, "selection")[: cfg.get("link_check_problems", 16)])
    with torch.no_grad():
        ce_old = target_ces(old_model, [r["full"] for r in rows], cfg["micro_batch"])
        ce_new = target_ces(new_model, [r["full"] for r in rows], cfg["micro_batch"])
    none = [abs(a - b) for r, a, b in zip(rows, ce_old, ce_new) if r["kind"] == "none"]
    hist = [b - a for r, a, b in zip(rows, ce_old, ce_new) if r["kind"] != "none"]
    if max(none) > 1e-6 or not others:
        raise RuntimeError(f"step-0 check failed: none-sample CE differs by {max(none)}, others identical={others}")
    dump(out / "start_manifest.json", {"step0.pt": sha(out / "step0.pt"),
                                       "model_config.json": sha(out / "model_config.json"),
                                       "reused_step0.pt": sha(reuse / "step0.pt")})
    dump(out / "link_complete.json", dict(
        source=str(cfg["reuse"]), problems=len(wanted), init_std=std, init_seed=int(cfg["history_init_seed"]),
        norms_before=column_norms(old_model), norms_after=column_norms(new_model), others_identical=bool(others),
        none_ce_max_abs_diff=float(max(none)), history_ce_mean_diff=float(np.mean(hist)) if hist else float("nan"),
        check_samples=dict(none=len(none), history=len(hist))))


def checkpoint_of(cfg: Any, out: Any, which: str) -> Path:
    if which == "r3_C":
        return ROOT / cfg["compare_reference"]
    return out / "train" / which / f"checkpoint_{cfg['compare_step']}.pt"


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["link", "train", "ablate", "copy-probe", "report"])
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--which", choices=WHICH)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    torch.set_num_threads(args.threads or cfg["threads"])
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "config.yaml"
    if frozen.exists() and yaml.safe_load(frozen.read_text()) != cfg:
        raise ValueError("config differs from the one this run started with")
    frozen.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    provenance(cfg, out, args.stage)
    start = time.perf_counter()

    if args.stage == "link":
        if not (out / "link_complete.json").exists():
            link(cfg, out)
        return
    if not (out / "link_complete.json").exists():
        raise RuntimeError("run the link stage first")
    m = load_step0(cfg, out)

    if args.stage == "train":
        from r4_samples import eval_loss, eval_set
        from train import train, warmup_constant
        from wb import Run

        g = args.group
        folder = out / "train" / g
        if (folder / "summary.json").exists():
            return
        fresh, _ = source(cfg, "fresh")
        sampler = sampler_for(cfg, out, m, g, fresh)
        evaluation = eval_set(sampler, ids(cfg, "selection"))
        extra = dict(group=g, kind_counts=sampler.counts, eval_dropped=sampler.eval_dropped)
        run = Run(cfg, out, f"train-{g}", "train", extra)

        def on_step(row: Any) -> None:
            run.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=row["step"])

        def on_checkpoint(step: Any) -> None:
            result = json.loads((folder / f"validation_{step}.json").read_text())
            run.log({f"eval/{k}": v for k, v in result.items() if isinstance(v, float)}, step=step)

        every, steps = cfg["checkpoint_every"], cfg["steps"]
        cooldown = (cfg["cooldown_from"], steps) if cfg["cooldown_from"] < steps else None
        saved = sorted(folder.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        train(cfg, out, m, sampler.wc, sampler.catalog, g, ids(cfg, "train"), ids(cfg, "selection"), steps, "train",
              schedule=warmup_constant(cfg["warmup"], cooldown),
              checkpoints=set(range(every, steps + 1, every)),
              resume_check_step=None, resume=saved[-1] if saved else None, folder=folder,
              on_checkpoint=on_checkpoint, on_step=on_step, batch_maker=sampler,
              validator=lambda step: eval_loss(m, evaluation, sampler.fractions, cfg["micro_batch"]))
        summary = json.loads((folder / "summary.json").read_text())
        run.summary(summary | {"usage": None})
        run.finish()
        return

    if args.stage in ("ablate", "copy-probe"):
        from wb import Run

        which = args.which
        ckpt = checkpoint_of(cfg, out, which)
        load_checkpoint(m, ckpt)
        m.eval()
        sampler = sampler_for(cfg, out, m, "IC")
        rel = str(ckpt.relative_to(ROOT))
        if args.stage == "ablate":
            from r4_ablate import CONDITIONS, ablate
            from r4_samples import eval_set

            target = out / "ablation" / f"{which}.json"
            if target.exists():
                return
            rows = eval_set(sampler, ids(cfg, "selection"), with_base=True)
            with torch.no_grad():
                result = ablate(m, rows, cfg["micro_batch"])
            dump(target, dict(checkpoint=rel, which=which, conditions=list(CONDITIONS),
                              dropped=sampler.eval_dropped, rows=result))
            run = Run(cfg, out, f"ablation-{which}", "ablation", dict(checkpoint=rel))
            groups: dict = {}
            for r in result:
                groups.setdefault(f"{r['kind']} b{r['bin']}", []).append(r)
            run.table("ablation/target_ce", ["group", "n", *CONDITIONS],
                      [[k, len(v), *(float(np.mean([r["ce"][c] for r in v])) for c in CONDITIONS)]
                       for k, v in sorted(groups.items())])
            run.summary({f"{k}/{c}": float(np.mean([r["ce"][c] for r in v])) for k, v in groups.items() for c in CONDITIONS})
            run.finish()
        else:
            from r4_ablate import copy_probe

            target = out / "copy_probe" / f"{which}.json"
            if target.exists():
                return
            pairs = []
            for pid in ids(cfg, "selection")[: cfg["copy_probe_problems"]]:
                seed = [eval_seed(cfg, pid), 7, 0]
                full = sampler.make("near", np.random.default_rng(seed), validation=True, pid=pid, level="exact")
                base = sampler.make("near", np.random.default_rng(seed), validation=True, pid=pid, level="exact",
                                    drop_history=True)
                if full["info"].get("pid") == pid and full["metadata"]["history_valid"]:
                    pairs.append((full, base))
            result = copy_probe(m, pairs)
            dump(target, dict(checkpoint=rel, which=which, result=result))
            run = Run(cfg, out, f"copy-probe-{which}", "copy_probe", dict(checkpoint=rel))
            run.summary({f"{k}/{key}": v for k, s in result.items() for key, v in s.items()})
            run.finish()
        print(f"{args.stage} {which} done in {time.perf_counter() - start:.1f}s")
        return

    if args.stage == "report":
        from r4_report import report

        report(cfg, out)
        return


if __name__ == "__main__":
    main()

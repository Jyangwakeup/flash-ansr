"""Near-correct-history small-scale run: A (no history) vs C (30% none / 40% near-correct /
30% real T8 candidates), 2,000 steps each (rules: 0-实验规则.md §12; procedure:
0-训练评估流程.md "第 3 轮：近似正确历史"). Every stage resumes.

    link                         problems 0..999 + validation problems, their model pools and
                                 step-0 from the round-2 run (hard links; re-drawn and compared)
    pools-train [--part i/n]     model pools of the new training problems (1000..4999)
    pools-selection              problems + model pools of the checkpoint-selection set
    pools-check                  every pool present -> pools_complete.json
    train --group A|C            2,000 steps, checkpoint + selection loss every 250
    select --group A|C           the pre-registered rule: lowest selection loss (ties: later step)
    self --condition NAME        full pipeline with the selected checkpoint (SELF_CONDITIONS)
    diagnose                     history-use diagnostic of C (diagnose.py --selected)
    report                       tables -> runs/.../报告.md (+ W&B)
"""

from pathlib import Path
from typing import Any
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))  # shared code: experiments/history_benefits/core
from common import ROOT, dump, problem_ids, read_problem, sha, source, worker_config  # noqa: E402
from run_support import jsonl_logger, load_checkpoint, load_step0, provenance  # noqa: E402

CONFIG = "experiments/history_benefits/r3_near_correct_history/config.yaml"
SELF_CONDITIONS = {"self_C_two": ("C", True), "self_C_direct": ("C", False), "self_A_direct": ("A", False)}


def ids(cfg: Any, split: str) -> list[int]:
    count = {"train": "train_problems", "validation": "validation_problems",
             "selection": "selection_problems"}[split]
    return problem_ids(cfg, split, cfg[count])


def link_file(src: Any, dst: Any) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def link(cfg: Any, out: Any) -> None:
    """Reuse the round-2 run's problems, pools and start; verify they are what the seeds give."""
    from pool import draw_problems

    reuse = ROOT / cfg["reuse"]
    old = yaml.safe_load((reuse / "config.yaml").read_text())
    for key in ("seed", "data_seed_train", "data_seed_validation", "fit_seed", "eval_seed",
                "recipe", "model", "draws", "restarts", "keep", "validation_sampling"):
        if old.get(key) != cfg.get(key) and not (key == "draws" and out.name.endswith("smoke")):
            raise ValueError(f"{key} differs from the reused run: {old.get(key)} vs {cfg.get(key)}")
    reused = ids(cfg, "train")[: cfg["reuse_train_problems"]] + ids(cfg, "validation")
    for pid in reused:
        for suffix in (".json", ".npz"):
            link_file(reuse / "problems" / f"{pid}{suffix}", out / "problems" / f"{pid}{suffix}")
            link_file(reuse / "pools" / "model" / f"{pid}{suffix}", out / "pools" / "model" / f"{pid}{suffix}")
    for name in ("step0.pt", "model_config.json"):
        if not (out / name).exists():
            shutil.copy2(reuse / name, out / name)
    manifest = json.loads((reuse / "start_manifest.json").read_text())
    for name, digest in manifest.items():
        if sha(out / name) != digest:
            raise ValueError(f"{name} differs from the round-2 run's start")
    dump(out / "start_manifest.json", manifest)
    # the seeds must reproduce the linked problems (prefix-stable streams)
    checked = {}
    for split, count in (("train", min(cfg["reuse_train_problems"], 50)), ("validation", len(ids(cfg, "validation")))):
        drawn = draw_problems(cfg, split, count)
        for pid, problem in zip(ids(cfg, split)[:count], drawn):
            if list(read_problem(out, pid).expression) != list(problem.expression):
                raise ValueError(f"problem {pid} does not match its seed")
        checked[split] = count
    dump(out / "link_complete.json", dict(reused=len(reused), redrawn_and_matched=checked,
                                          source=str(cfg["reuse"])))


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["link", "pools-train", "pools-selection", "pools-check", "train",
                                          "select", "self", "diagnose", "report"])
    parser.add_argument("--group", choices=["A", "C"])
    parser.add_argument("--condition", choices=sorted(SELF_CONDITIONS))
    parser.add_argument("--part", default="0/1", help="pools-train: lane i of n (i/n)")
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

    if args.stage in ("pools-train", "pools-selection"):
        from pool import build

        if args.stage == "pools-train":
            i, n = map(int, args.part.split("/"))
            todo = ids(cfg, "train")[cfg["reuse_train_problems"]:][i::n]
        else:
            todo = ids(cfg, "selection")
        build(cfg, out, todo, kinds=("model",))
        return
    if args.stage == "pools-check":
        wanted = ids(cfg, "train") + ids(cfg, "validation") + ids(cfg, "selection")
        missing = [p for p in wanted if not (out / "pools" / "model" / f"{p}.json").exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} pools missing, e.g. {missing[:5]}")
        dump(out / "pools_complete.json", dict(n=len(wanted), checked_at=time.strftime("%F %T")))
        return
    if not (out / "pools_complete.json").exists():
        raise RuntimeError("run pools-check first")
    m = load_step0(cfg, out)

    if args.stage == "train":
        from r3_samples import MixSampler, selection_loss, selection_set
        from train import train, warmup_constant
        from wb import Run

        g = args.group
        folder = out / "train" / g
        if (folder / "summary.json").exists():
            return
        src, recipe = source(cfg)
        wc = worker_config(cfg, m, recipe)
        fresh, _ = source(cfg, "fresh")
        sampler = MixSampler(cfg, out, m, wc, src.catalog, fresh, g, ids(cfg, "train"))
        selection = selection_set(sampler, ids(cfg, "selection"))
        run = Run(cfg, out, f"train-{g}", "train", dict(group=g, kind_counts=sampler.counts))

        def on_step(row: Any) -> None:
            run.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=row["step"])

        def on_checkpoint(step: Any) -> None:
            result = json.loads((folder / f"validation_{step}.json").read_text())
            run.log({f"selection/{k}": v for k, v in result.items() if isinstance(v, float)}, step=step)

        every = cfg["checkpoint_every"]
        saved = sorted(folder.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        train(cfg, out, m, wc, src.catalog, g, ids(cfg, "train"), ids(cfg, "selection"), cfg["steps"], "train",
              schedule=warmup_constant(cfg["warmup"], (cfg["cooldown_from"], cfg["steps"])),
              checkpoints=set(range(every, cfg["steps"] + 1, every)),
              resume_check_step=None, resume=saved[-1] if saved else None, folder=folder,
              on_checkpoint=on_checkpoint, on_step=on_step, batch_maker=sampler,
              validator=lambda step: selection_loss(m, selection, sampler.fractions, cfg["micro_batch"]))
        summary = json.loads((folder / "summary.json").read_text())
        run.summary(summary | {"usage": None})
        run.finish()
        return

    if args.stage == "select":
        g = args.group
        folder = out / "train" / g
        rows = []
        for path in folder.glob("validation_*.json"):
            step = int(path.stem.split("_")[1])
            if step > 0:
                rows.append((json.loads(path.read_text())["loss"], step))
        if len(rows) != cfg["steps"] // cfg["checkpoint_every"]:
            raise RuntimeError(f"expected {cfg['steps'] // cfg['checkpoint_every']} checkpoints, got {len(rows)}")
        best_loss = min(v for v, _ in rows)
        step = max(s for v, s in rows if v == best_loss)  # ties: the later step
        ckpt = folder / f"checkpoint_{step}.pt"
        dump(folder / "selected.json", dict(
            group=g, step=step, loss=best_loss, checkpoint=str(ckpt.relative_to(ROOT)),
            rule="lowest selection loss over saved checkpoints (step > 0); ties: the later step",
            losses={str(s): v for v, s in sorted(rows, key=lambda r: r[1])}))
        return

    if args.stage == "self":
        from evaluation import run_self, summarize
        from wb import Run

        who, two = SELF_CONDITIONS[args.condition]
        selected = json.loads((out / "train" / who / "selected.json").read_text())
        load_checkpoint(m, ROOT / selected["checkpoint"])
        final = out / "final"
        final.mkdir(exist_ok=True)
        name = args.condition
        done = final / f"{name}.json"
        if done.exists():
            return
        log = final / f"{name}.jsonl"
        seen = [json.loads(line) for line in log.open()] if log.exists() else []
        todo = [p for p in ids(cfg, "validation") if p not in {r["pid"] for r in seen}]
        run = Run(cfg, out, name, "self", dict(condition=name, selected_step=selected["step"]))
        to_file = jsonl_logger(log)

        def logged(row: Any) -> None:
            to_file(row)
            seen.append(row)
            values = {f"running/{k}": v for k, v in summarize(seen).items()}
            values["problem/seconds"] = row["seconds_total"]
            run.log(values, step=len(seen))

        try:
            run_self(cfg, out, m, todo, two, log=logged)
            rows = [json.loads(line) for line in log.open()]
            summary = summarize(rows)
            dump(done, dict(name=name, selected=selected, summary=summary, rows=rows))
            run.summary(summary)
        finally:
            run.finish()
        return

    if args.stage == "diagnose":
        if (out / "diagnostics" / "history_use.json").exists():
            return
        subprocess.run([sys.executable, str(ROOT / "experiments/history_benefits/core/diagnose.py"),
                        "--config", args.config, "--groups", "C", "--selected",
                        "--n", str(cfg["diagnose_problems"]), "--draws", str(cfg["diagnose_draws"])],
                       check=True, cwd=ROOT / "experiments/history_benefits/core")
        return

    if args.stage == "report":
        from r3_report import report

        report(cfg, out)
        return
    print(f"{args.stage} done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()

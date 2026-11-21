"""Medium-scale history-refinement run (rules: experiment-results/history-benefits/正式实验规则.md;
procedure: 训练评估流程.md). Every stage resumes: a finished artifact is never rebuilt.

Stages (independent ones may run as parallel processes):

    pools-train | pools-validation        build problems + candidate pools
    step0                                  the shared start (after pools)
    train --group A|B|C                    main line, 1,000 steps, checkpoint + probe every 200
    cooldown --group A|B|C                 branch from step 800, lr -> 0 at step 1,000
    probe-main --group A|B|C               small probe on the main-line step-1,000 (reference)
    final --condition NAME                 one of the six final conditions (cooldown step 1,000)
    original                               frozen T8, 1,000 draws (same-draws baseline)
    step0-eval                             step-0, no history, 1,000 draws (transfer control)
    equal-time-measure                     time one sample of each method -- run ALONE (timing)
    equal-time --condition NAME|original_T8  rerun at the time-matched budget (parallel is fine)
    self --condition NAME                  full pipeline, the model alone (规则 §2.3): NAME in
                                           SELF_CONDITIONS (added after the final results)
    report                                 tables and the medium-scale report

Timing caveat: stages that run in parallel contend for the CPU, so their seconds are only
comparable within one stage; the equal-time stage measures its own timing and must run alone.
"""

from typing import Any
import argparse
import json
import subprocess
import sys
import time

import torch
import yaml

from common import ROOT, dump, migrate, load_base, problem_ids, sha, source, worker_config
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from pool import build

CONFIG = "experimental/history_benefits/medium.yaml"
GROUP_SOURCE = {"A": "model", "B": "random", "C": "model"}  # A never uses its pool
#: full-pipeline conditions: name -> (model, two_stage). "T8" is the frozen original model run
#: through the same hand-written inference code as A/B/C.
SELF_CONDITIONS = {
    "self_C_two": ("C", True), "self_C_direct": ("C", False),
    "self_B_two": ("B", True), "self_B_direct": ("B", False),
    "self_A_direct": ("A", False), "self_T8_direct": ("T8", False),
}


def provenance(cfg: Any, out: Any, stage: Any) -> None:
    """Record config, code version and file hashes of one stage (append-only)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa: E731
    files = sorted((ROOT / "experiments/history_benefits").glob("*.py"))
    record = dict(
        stage=stage,
        time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        argv=sys.argv,
        git_commit=git("rev-parse", "HEAD").strip(),
        git_dirty=bool(git("status", "--porcelain").strip()),
        hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
        torch=torch.__version__,
        threads=torch.get_num_threads(),
    )
    with (out / "provenance.jsonl").open("a") as log:
        log.write(json.dumps(record) + "\n")


def load_step0(cfg: Any, out: Any) -> Any:
    m = FlashANSRModel.from_config(json.loads((out / "model_config.json").read_text()))
    m.load_state_dict(torch.load(out / "step0.pt", weights_only=True))
    return m


def load_checkpoint(m: Any, path: Any) -> Any:
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return m


def ids_of(cfg: Any) -> tuple[list[int], list[int]]:
    return (problem_ids(cfg, "train", cfg["train_problems"]),
            problem_ids(cfg, "validation", cfg["validation_problems"]))


def jsonl_logger(path: Any) -> Any:
    def log(row: Any) -> None:
        with path.open("a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(f"{path.parent.name}/{path.stem} pid={row['pid']}", flush=True)
    return log


def probe(cfg: Any, out: Any, m: Any, group: Any, folder: Any, step: Any) -> None:
    """Every-200-step probe (流程 §6): history off, and history on for B/C; probe_draws each."""
    from medium_eval import run_condition, summarize

    path = folder / f"probe_{step}.json"
    if path.exists():
        return
    _, val = ids_of(cfg)
    sources = [None] if group == "A" else [None, GROUP_SOURCE[group]]
    result = {}
    for src in sources:
        rows = run_condition(cfg, out, m, val, src, cfg["probe_draws"], phase0=False)
        result[str(src)] = dict(phase1=summarize(rows, "phase1"), rows=rows)
    dump(path, dict(step=step, group=group, **result))
    m.train()


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["pools-train", "pools-validation", "step0", "train",
                                          "cooldown", "probe-main", "final", "original",
                                          "step0-eval", "equal-time-measure", "equal-time", "self", "report"])
    parser.add_argument("--group", choices=["A", "B", "C"])
    parser.add_argument("--condition")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--threads", type=int, help="override torch threads for this process")
    parser.add_argument("--kind", choices=["random", "model"],
                        help="pools: build only this kind, last problem first (a helper process "
                             "next to the main pool builder; pools are deterministic, so a "
                             "problem both happen to build is simply written twice)")
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
    train_ids, val_ids = ids_of(cfg)
    start = time.perf_counter()

    if args.stage in ("pools-train", "pools-validation"):
        ids = train_ids if args.stage == "pools-train" else val_ids
        if args.kind:
            build(cfg, out, ids[::-1], kinds=(args.kind,))
            return
        build(cfg, out, ids)
        dump(out / f"{args.stage}_complete.json", dict(seconds=time.perf_counter() - start))
        return

    if args.stage == "step0":
        if not (out / "step0.pt").exists():
            base = load_base(cfg)
            try:
                m, mc = migrate(cfg, base.flash_ansr_model)
            finally:
                base.close()
            torch.save(m.state_dict(), out / "step0.pt")
            dump(out / "model_config.json", mc)
        dump(out / "start_manifest.json",
             {n: sha(out / n) for n in ("step0.pt", "model_config.json")})
        return

    for name in ("pools-train_complete.json", "pools-validation_complete.json", "step0.pt"):
        if not (out / name).exists():
            raise RuntimeError(f"missing {name}: run the earlier stages first")
    m = load_step0(cfg, out)

    if args.stage in ("train", "cooldown"):
        from train import train, warmup_constant

        g = args.group
        folder = out / "main" / g if args.stage == "train" else out / "cooldown" / g
        if (folder / "summary.json").exists():
            return
        src, recipe = source(cfg)
        wc = worker_config(cfg, m, recipe)
        every = cfg["checkpoint_every"]
        from wb import Run

        run = Run(cfg, out, f"{args.stage}-{g}", args.stage, dict(group=g))

        def on_step(row: Any) -> None:
            run.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=row["step"])

        def on_checkpoint(step: Any) -> None:
            probe(cfg, out, m, g, folder, step)
            values = {}
            val = folder / f"validation_{step}.json"
            if val.exists():
                values["val/loss"] = json.loads(val.read_text())["loss"]
            result = json.loads((folder / f"probe_{step}.json").read_text())
            for key, part in result.items():
                if isinstance(part, dict) and "phase1" in part:
                    label = "no_history" if key == "None" else f"{key}_history"
                    for metric in ("vNRR", "fNRR", "oracle_val"):
                        values[f"probe/{label}/{metric}"] = part["phase1"][metric]
            run.log(values, step=step)
        if args.stage == "train":
            saved = sorted(folder.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            resume = saved[-1] if saved else None  # an interrupted main line continues
            train(cfg, out, m, wc, src.catalog, g, train_ids, val_ids, cfg["steps"], "main",
                  schedule=warmup_constant(cfg["warmup"]),
                  checkpoints=set(range(every, cfg["steps"] + 1, every)),
                  resume_check_step=every if resume is None else None, resume=resume,
                  folder=folder, on_checkpoint=on_checkpoint, on_step=on_step)
        else:
            branch = out / "main" / g / f"checkpoint_{cfg['cooldown_from']}.pt"
            train(cfg, out, m, wc, src.catalog, g, train_ids, val_ids, cfg["steps"], "cooldown",
                  schedule=warmup_constant(cfg["warmup"], (cfg["cooldown_from"], cfg["steps"])),
                  checkpoints={cfg["steps"]}, resume_check_step=None, resume=branch,
                  folder=folder, on_checkpoint=on_checkpoint, on_step=on_step)
        run.summary(json.loads((folder / "summary.json").read_text()) | {"usage": None})
        run.finish()
        return

    if args.stage == "probe-main":
        # the main line's final step already got its probe during training (on_checkpoint)
        return

    from medium_eval import CONDITIONS, run_condition, run_original, summarize

    final = out / "final"
    final.mkdir(exist_ok=True)

    def run_rows(name: Any, fn: Any) -> None:
        done = final / f"{name}.json"
        if done.exists():
            return
        log = final / f"{name}.jsonl"
        seen = [json.loads(l) for l in log.open()] if log.exists() else []
        finished = {r["pid"] for r in seen}
        rows_ids = [p for p in val_ids if p not in finished]
        from wb import Run

        run = Run(cfg, out, name, args.stage, dict(condition=name, problems=len(val_ids)))
        to_file = jsonl_logger(log)

        def logged(row: Any) -> None:  # file first (the record), then the running rates on W&B
            to_file(row)
            seen.append(row)
            values = {f"running/{k}": v for k, v in summarize(seen).items()}
            values["problem/seconds"] = row["seconds_total"]
            values["problem/n_candidates"] = row["merged"]["n_candidates"]
            run.log(values, step=len(seen))

        try:
            fn(rows_ids, logged)
            rows = [json.loads(l) for l in log.open()]
            summary = summarize(rows)
            dump(done, dict(name=name, summary=summary, rows=rows))
            run.summary(summary | ({"phase1": summarize(rows, "phase1")} if rows and rows[0].get("phase1") else {}))
        finally:
            run.finish()

    if args.stage == "final":
        group, hist = CONDITIONS[args.condition]
        load_checkpoint(m, out / "cooldown" / group / f"checkpoint_{cfg['steps']}.pt")
        run_rows(args.condition, lambda ids, log: run_condition(
            cfg, out, m, ids, hist, cfg["phase1_draws"], phase0=True, log=log))
        return
    if args.stage == "original":
        run_rows("original_T8", lambda ids, log: run_original(cfg, out, ids, cfg["baseline_draws"], log=log))
        return
    if args.stage == "step0-eval":
        run_rows("step0_none", lambda ids, log: run_condition(
            cfg, out, m, ids, None, cfg["baseline_draws"], phase0=False, log=log))
        return
    if args.stage == "equal-time-measure":
        from equal_time import measure
        from wb import Run

        if (out / "equal_time_budget.json").exists():
            return
        measure(cfg, out, m, val_ids)
        budget = json.loads((out / "equal_time_budget.json").read_text())
        run = Run(cfg, out, "equal-time-measure", args.stage)
        run.summary({k: v for k, v in budget.items() if k not in ("timed_problems", "note")})
        run.table("equal_time/draws", ["condition", "seconds_per_draw", "phase1_draws"],
                  [[k, budget["phase1_seconds_per_draw"][k], budget["phase1_draws"][k]]
                   for k in budget["phase1_draws"]])
        run.finish()
        return
    if args.stage == "equal-time":
        budget = json.loads((out / "equal_time_budget.json").read_text())
        name = args.condition
        if name == "original_T8":
            n = budget["original_draws"]
            run_rows(f"equal_time_{name}", lambda ids, log: run_original(cfg, out, ids, n, log=log))
            return
        group, hist = CONDITIONS[name]
        load_checkpoint(m, out / "cooldown" / group / f"checkpoint_{cfg['steps']}.pt")
        n = budget["phase1_draws"][name]
        run_rows(f"equal_time_{name}", lambda ids, log: run_condition(
            cfg, out, m, ids, hist, n, phase0=True, log=log))
        return
    if args.stage == "self":
        from medium_eval import run_self

        who, two = SELF_CONDITIONS[args.condition]
        if who == "T8":
            base = load_base(cfg)
            model = base.flash_ansr_model
        else:
            model = load_checkpoint(m, out / "cooldown" / who / f"checkpoint_{cfg['steps']}.pt")
        run_rows(args.condition, lambda ids, log: run_self(cfg, out, model, ids, two, log=log))
        return
    if args.stage == "report":
        from medium_report import report
        report(cfg, out)
        return


if __name__ == "__main__":
    main()

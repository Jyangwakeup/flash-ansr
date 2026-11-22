"""比较原 T8、修复后的 step-0、A 组训练 100 步后的模型（同题、同生成次数、同打分）。

分析报告：experiment-results/T8vsT8-1/迁移对照.md

用法（在仓库根目录）：
  .venv/bin/python experiments/T8vsT8-1/run.py smoke     # 只跑 160、161，并做复制版自检
  .venv/bin/python experiments/T8vsT8-1/run.py full      # 跑全部 40 道验证题
  .venv/bin/python experiments/T8vsT8-1/run.py summarize # 只汇总已有结果

不改动 experiments/history_benefits 下的代码，也不写入 runs/history_benefits/pilot；结果保存在 runs/T8vsT8-1/。
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/history_benefits"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp_wandb import Run  # noqa: E402
from common import dump, load_base, read_problem, seed_all  # noqa: E402
from evaluate import evaluate  # noqa: E402
from pool import result_rows  # noqa: E402
from flash_ansr import FlashANSR, SoftmaxSamplingConfig  # noqa: E402
from flash_ansr.flash_ansr import Generation  # noqa: E402
from flash_ansr.model.flash_ansr_model import FlashANSRModel  # noqa: E402
from flash_ansr.preprocessing import PromptPrefix  # noqa: E402

CONFIG = ROOT / "experimental/history_benefits/pilot.yaml"
PILOT = ROOT / "runs/history_benefits/pilot"
RESULTS = ROOT / "runs/T8vsT8-1"
VALIDATION = list(range(160, 200))
THRESHOLDS = (1e-3, 1e-6)


def workdir() -> Path:
    """results/ 里用链接指向试跑的题目和候选池，原有评估函数就能直接读。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ("problems", "pools"):
        link = RESULTS / name
        if not link.exists():
            link.symlink_to(PILOT / name, target_is_directory=True)
    return RESULTS


def load_step0() -> FlashANSRModel:
    m = FlashANSRModel.from_config(json.loads((PILOT / "model_config.json").read_text()))
    m.load_state_dict(torch.load(PILOT / "step0.pt", weights_only=True))
    return m


def evaluate_no_history(cfg: Any, out: Path, model: Any, ids: Any, label: str) -> Any:
    """evaluate.py 中 K=0（不给历史）那条路的逐行复制。

    唯一区别：模型没有历史位置时（原 T8），只传 (x, y) 两部分数据，不传 history_mask。
    """
    has_slot = getattr(model, "history_slots", 0) == 1
    wrapper = FlashANSR(
        model.simplipy_engine,
        model,
        model.tokenizer,
        generation_config=SoftmaxSamplingConfig(draws=cfg["evaluation_draws"], emission="fittable"),
        refine={"n_restarts": cfg["restarts"]},
        ranking="mdl",
        compute={"device": "cpu", "workers": cfg["workers"]},
    )
    path = out / label / "evaluation.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["pid"] for r in rows}
    if set(ids) <= done:
        return rows
    run = Run("T8vsT8-1", name=label, job_type="eval", config=dict(cfg, model=label, problems=list(ids)), out=out)
    model.eval()
    tok = model.tokenizer
    try:
        for pid in ids:
            if pid in done:
                continue
            p = read_problem(out, pid)
            prefix = [tok["<bos>"], tok["<mask_fittable>"], tok["<expression>"]]
            columns = [p.x_support, p.y_support_noisy]
            if has_slot:
                columns.append(np.zeros(len(p.x_support)))
            data = torch.tensor(np.column_stack(columns), dtype=torch.float64).unsqueeze(0)
            seed_all(cfg["seed"] + pid)
            t0 = time.perf_counter()
            prompt = PromptPrefix(prefix, [float("nan")] * len(prefix), [True] * len(prefix), {})
            with torch.no_grad():
                if has_slot:
                    memory = model._create_memory(data, history_mask=torch.tensor([False]))
                else:
                    memory = model._create_memory(data)
                beams, logs = model.sample_top_kp(
                    data,
                    draws=cfg["evaluation_draws"],
                    max_len=len(prefix) + cfg["generation_tokens"],
                    batch_size=cfg["evaluation_draws"],
                    return_raw=True,
                    static_decode=False,
                    use_cache=True,
                    prompt_prefix=prompt,
                    memory=memory,
                )
            accepted, rejections = [], []
            for index, (beam, logprob) in enumerate(zip(beams, logs)):
                try:
                    tok.extract_expression_from_beam(beam)
                except (ValueError, IndexError) as exc:
                    rejections.append(dict(index=index, reason=str(exc)))
                    continue
                accepted.append((beam, logprob))
            gen = Generation(
                [b for b, _ in accepted],
                [v for _, v in accepted],
                p.x_support,
                p.y_support_noisy,
                float(np.var(p.y_support_noisy)),
                len(p.x_support),
                prompt,
                memory,
                torch.device("cpu"),
                {},
                None,
                time.perf_counter() - t0,
                cfg["evaluation_draws"],
                cfg["seed"] + pid,
            )
            outcome = wrapper._fit_refine(gen, refine_seed=cfg["seed"] + pid, allow_empty=True)
            result = wrapper._build_result(gen, outcome)
            candidates = [r for r, _ in result_rows(result, p.x_support, p.y_support_noisy, "model")]
            rows.append(
                dict(
                    pid=pid,
                    model=label,
                    raw_count=len(beams),
                    extraction_failures=rejections,
                    candidates=candidates,
                    seconds=time.perf_counter() - t0,
                )
            )
            dump(path, rows)
            print(f"{label}/{pid}: {len(candidates)} 个可用公式", flush=True)
            pp = per_problem(rows[-1])
            run.log({"problem/pid": pid, "problem/n_candidates": pp["n"],
                     "problem/log10_best_fvu": float(np.log10(max(pp["best"], 1e-16))),
                     "problem/seconds": rows[-1]["seconds"],
                     "running/solved_best<1e-3": float(np.mean([per_problem(r)["best"] < 1e-3 for r in rows]))},
                    step=len(rows))
        run.summary(stats([per_problem(r) for r in rows if r["pid"] in ids]))
    finally:
        wrapper.close()
        run.finish()
    return rows


def self_check(cfg: Any, out: Path, ids: Any) -> Any:
    """复制版和原版 evaluate 各测一次 step-0，可用公式必须完全一样。"""
    original = evaluate(cfg, out, load_step0(), "A", ids, "selfcheck")
    copied = evaluate_no_history(cfg, out, load_step0(), ids, "selfcheck_copy")
    key = lambda c: (tuple(c["expression_prefix"]), c["fvu"], c["score"])  # noqa: E731
    report = []
    for a, b in zip(sorted(original, key=lambda r: r["pid"]), sorted(copied, key=lambda r: r["pid"])):
        same = [key(c) for c in a["candidates"]] == [key(c) for c in b["candidates"]]
        report.append(dict(pid=a["pid"], same=same, original=len(a["candidates"]), copy=len(b["candidates"])))
    dump(out / "selfcheck.json", report)
    if not all(r["same"] for r in report):
        raise AssertionError(f"复制版与原版结果不一致：{report}")
    print("自检通过：复制版与原版结果一致", flush=True)


# ---------------- 汇总 ----------------

def per_problem(row: dict) -> dict:
    fvus = [c["fvu"] for c in row["candidates"] if c["fvu"] is not None and np.isfinite(c["fvu"])]
    ranked = sorted(
        (c for c in row["candidates"] if c["score"] is not None), key=lambda c: c["score"]
    )
    best = min(fvus) if fvus else float("inf")
    top = ranked[0]["fvu"] if ranked and ranked[0]["fvu"] is not None else float("inf")
    return dict(pid=row["pid"], n=len(row["candidates"]), best=best, top=top,
                extraction_failures=len(row.get("extraction_failures", [])))


def stats(problems: Any) -> dict:
    out = dict(
        problems=len(problems),
        failed=sum(p["n"] == 0 for p in problems),
        median_log10_best=float(np.median([np.log10(max(p["best"], 1e-16)) for p in problems])),
        extraction_failures=sum(p["extraction_failures"] for p in problems),
    )
    for t in THRESHOLDS:
        out[f"solved_best<{t:g}"] = float(np.mean([p["best"] < t for p in problems]))
        out[f"solved_top<{t:g}"] = float(np.mean([p["top"] < t for p in problems]))
    return out


def paired(a: Any, b: Any, name_a: str, name_b: str, seed: int = 0) -> dict:
    """同题比较最佳误差：log10 相差超过 0.1（约 25%）才算一方更好；两边都 <1e-6 算打平。"""
    a = {p["pid"]: p for p in a}
    b = {p["pid"]: p for p in b}
    pids = sorted(set(a) & set(b))
    la = np.array([np.log10(max(a[i]["best"], 1e-16)) for i in pids])
    lb = np.array([np.log10(max(b[i]["best"], 1e-16)) for i in pids])
    both_exact = (la < -6) & (lb < -6)
    diff = np.where(both_exact, 0.0, lb - la)
    sa = np.array([a[i]["best"] < 1e-3 for i in pids], dtype=float)
    sb = np.array([b[i]["best"] < 1e-3 for i in pids], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [np.mean((sb - sa)[rng.integers(0, len(pids), len(pids))]) for _ in range(5000)]
    return dict(
        comparison=f"{name_b} 对比 {name_a}",
        problems=len(pids),
        better=int(np.sum(diff < -0.1)),
        worse=int(np.sum(diff > 0.1)),
        tie=int(np.sum(np.abs(diff) <= 0.1)),
        solved_rate_change=float(np.mean(sb - sa)),
        solved_rate_change_95ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        median_log10_change=float(np.median(diff)),
    )


def summarize(ids: Any) -> Any:
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    pick = lambda rows, k=None: [per_problem(r) for r in rows if r["pid"] in ids and (k is None or r["requested_history"] == k)]  # noqa: E731
    models = {
        "原T8": pick(load(RESULTS / "T8/evaluation.json")),
        "step-0": pick(load(RESULTS / "step0/evaluation.json")),
        "A@100": pick(load(PILOT / "pilot/A/evaluation.json")),
    }
    history = {}
    for g in ("B", "C"):
        rows = load(PILOT / f"pilot/{g}/evaluation.json")
        history[g] = dict(K0=stats(pick(rows, 0)), K1=stats(pick(rows, 1)),
                          paired=paired(pick(rows, 0), pick(rows, 1), f"{g} 不给历史", f"{g} 给历史"))
    summary = dict(
        problems=ids,
        draws_per_problem=yaml.safe_load(CONFIG.read_text())["evaluation_draws"],
        models={k: stats(v) for k, v in models.items()},
        paired=[
            paired(models["原T8"], models["step-0"], "原T8", "step-0"),
            paired(models["step-0"], models["A@100"], "step-0", "A@100"),
            paired(models["原T8"], models["A@100"], "原T8", "A@100"),
        ],
        history=history,
        per_problem={k: v for k, v in models.items()},
        note="只有 40 道题、每题 20 次生成：只看方向，不作结论。",
    )
    dump(RESULTS / "summary.json", summary)
    with Run("T8vsT8-1", name=f"summary-{len(ids)}", job_type="summary",
             config=dict(problems=ids, draws_per_problem=summary["draws_per_problem"]), out=RESULTS) as run:
        cols = sorted({k for v in summary["models"].values() for k in v})
        run.table("models", ["model"] + cols, [[m] + [v.get(c) for c in cols] for m, v in summary["models"].items()])
        pcols = ["comparison", "problems", "better", "worse", "tie", "solved_rate_change", "median_log10_change"]
        run.table("paired", pcols, [[p[c] for c in pcols] for p in summary["paired"]])
        run.summary({f"{m}/{k}": v for m, s in summary["models"].items() for k, v in s.items()})
    print(json.dumps({k: summary[k] for k in ("models", "paired", "history")}, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "full", "summarize"])
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    torch.set_num_threads(cfg["threads"])
    out = workdir()
    ids = [160, 161] if args.stage == "smoke" else VALIDATION
    if args.stage != "summarize":
        if args.stage == "smoke":
            self_check(cfg, out, ids)
        base = load_base(cfg)
        try:
            evaluate_no_history(cfg, out, base.flash_ansr_model, ids, "T8")
        finally:
            base.close()
        evaluate_no_history(cfg, out, load_step0(), ids, "step0")
    summarize(ids)


if __name__ == "__main__":
    main()

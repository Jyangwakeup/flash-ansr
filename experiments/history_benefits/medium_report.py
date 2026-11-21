"""Medium-scale report (流程 §8, 规则 §6): tables, paired bootstrap CIs, curves; also to W&B."""

from typing import Any
import json
from pathlib import Path

import numpy as np

from common import ROOT, train_seed
from medium_eval import CONDITIONS

N_BOOT = 10_000
H4_TOLERANCE = 0.03
LABELS = {
    "1_A_none": "① A 不给历史",
    "2_C_model": "② C 给模型历史",
    "3_B_random": "③ B 给随机历史",
    "4_C_none": "④ C 不给历史",
    "5_B_none": "⑤ B 不给历史",
    "6_B_model": "⑥ B 给模型历史",
    "original_T8": "原 T8 直接生成 1000 个",
    "step0_none": "step-0 不给历史，生成 1000 个",
}
COMPARISONS = [
    ("2_C_model", "1_A_none", "C 方案整体 vs 只多训练"),
    ("3_B_random", "1_A_none", "B 方案整体 vs 只多训练"),
    ("2_C_model", "4_C_none", "C 模型：历史本身有没有用"),
    ("3_B_random", "5_B_none", "B 模型：历史本身有没有用"),
    ("6_B_model", "2_C_model", "H4：同样给模型历史，B − C"),
    ("2_C_model", "original_T8", "C 方案 vs 原 T8（同样 1000 个）"),
    ("3_B_random", "original_T8", "B 方案 vs 原 T8（同样 1000 个）"),
    ("1_A_none", "original_T8", "A vs 原 T8（同样 1000 个）"),
    ("step0_none", "original_T8", "迁移对照：step-0 vs 原 T8"),
]


def load(out: Any, name: Any) -> Any:
    path = out / "final" / f"{name}.json"
    return json.loads(path.read_text())["rows"] if path.exists() else None


def solved(rows: Any, key: Any = "top1_val") -> dict:
    return {r["pid"]: bool(r["merged"] and r["merged"][key]) for r in rows}


def paired(a: Any, b: Any, seed: Any) -> dict:
    """Paired difference of success rates (a − b) over common problems, bootstrap 95% CI."""
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1) if len(d) else np.array([np.nan])
    return dict(n=len(ids), diff=float(d.mean()) if len(d) else float("nan"),
                lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)))


def rate(rows: Any, part: Any, key: Any) -> float:
    return sum(bool(r[part] and r[part][key]) for r in rows) / len(rows)


def report(cfg: Any, out: Any) -> None:
    from wb import Run

    names = list(CONDITIONS) + ["original_T8", "step0_none"]
    data = {n: load(out, n) for n in names}
    lines = ["# 中等规模报告", "",
             f"产物：`{cfg['output']}`（配置副本：`{cfg['output']}/config.yaml`）。",
             "规则：[正式实验规则.md](正式实验规则.md)；流程：[训练评估流程.md](训练评估流程.md)。", "",
             "**中等规模的结果只用来判断方向，不当作正式结论。** 主结果为收尾支线第 1,000 步；",
             "解对 = 留出点 FVU ≤ float32 eps；所有题都在分母里，没有答案记为没解对。", "",
             "## 主表（同样 1000 个预算）", "",
             "| 条件 | 题数 | vNRR | fNRR | Oracle（留出点） | 第二阶段自身 vNRR | 第一阶段未解出的题中新解出 | 平均秒数/题 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    table = []
    for n in names:
        rows = data[n]
        if rows is None:
            lines.append(f"| {LABELS[n]} | — | 未完成 | | | | | |")
            continue
        p1 = rate(rows, "phase1", "top1_val") if rows[0].get("phase1") else float("nan")
        unsolved = [r for r in rows if r.get("phase0") and not r["phase0"]["oracle_val"]]
        new = (f"{sum(r['phase1']['oracle_val'] for r in unsolved)}/{len(unsolved)}"
               if unsolved else "不适用")
        v, f, o = rate(rows, "merged", "top1_val"), rate(rows, "merged", "top1_fit"), rate(rows, "merged", "oracle_val")
        sec = float(np.mean([r["seconds_total"] for r in rows]))
        lines.append(f"| {LABELS[n]} | {len(rows)} | {v:.1%} | {f:.1%} | {o:.1%} | "
                     f"{'' if p1 != p1 else f'{p1:.1%}'} | {new} | {sec:.1f} |")
        table.append([LABELS[n], len(rows), v, f, o, p1, sec])
    lines += ["", "秒数在多个实验并行时测得，只能在同一阶段内部比较；时间比较以同时间对比为准。", "",
              "## 配对比较（vNRR 差值，bootstrap 10,000 次，95% 置信区间）", "",
              "| 比较 | 题数 | 差值 | 95% 置信区间 |", "|---|---:|---:|---|"]
    comps = []
    for a, b, label in COMPARISONS:
        if data.get(a) is None or data.get(b) is None:
            lines.append(f"| {label} | — | 未完成 | |")
            continue
        c = paired(solved(data[a]), solved(data[b]), train_seed(cfg))
        note = ""
        if a == "6_B_model":
            note = "（达到「效果接近」）" if c["lo"] >= -H4_TOLERANCE else "（未达到「效果接近」：下限低于 −3 个百分点）"
        lines.append(f"| {label} | {c['n']} | {c['diff']:+.1%} | [{c['lo']:+.1%}, {c['hi']:+.1%}]{note} |")
        comps.append([label, c["n"], c["diff"], c["lo"], c["hi"]])
    lines += ["", "## 建池成本（H4，规则第 6 节）", "",
              "| 池 | 题数 | 总秒数 | 每题秒数 | 平均保留候选数 |", "|---|---:|---:|---:|---:|"]
    pool_cost = {}
    for kind in ("random", "model"):
        rows = [json.loads(f.read_text()) for f in (out / "pools" / kind).glob("*.json")]
        if rows:
            total = sum(r["seconds"] for r in rows)
            pool_cost[kind] = total
            lines.append(f"| {kind} | {len(rows)} | {total:.0f} | {total / len(rows):.1f} | "
                         f"{np.mean([len(r['candidates']) for r in rows]):.1f} |")
    if len(pool_cost) == 2:
        lines += ["", f"随机池耗时是模型池的 {pool_cost['random'] / pool_cost['model']:.0%}。两个池同时建、"
                  "和其他任务抢 CPU，比例只作参考。"]
    budget = out / "equal_time_budget.json"
    if budget.exists():
        b = json.loads(budget.read_text())
        lines += ["", "## 同时间对比", "",
                  f"每题时间预算 {b['budget_seconds_per_problem']:.1f} 秒（原 T8 生成 1000 个的中位数，"
                  f"单独测量）；第一阶段 {b['phase0_seconds_per_problem']:.1f} 秒。", "",
                  "| 条件 | 第二阶段生成数 | vNRR | 与原 T8 的差（95% CI） |", "|---|---:|---:|---|"]
        base = load(out, "equal_time_original_T8") or data.get("original_T8")
        for n in CONDITIONS:
            rows = load(out, f"equal_time_{n}")
            if rows is None or base is None:
                lines.append(f"| {LABELS[n]} | {b['phase1_draws'][n]} | 未完成 | |")
                continue
            c = paired(solved(rows), solved(base), train_seed(cfg))
            lines.append(f"| {LABELS[n]} | {b['phase1_draws'][n]} | {rate(rows, 'merged', 'top1_val'):.1%} | "
                         f"{c['diff']:+.1%} [{c['lo']:+.1%}, {c['hi']:+.1%}] |")
    else:
        lines += ["", "## 同时间对比", "", "未完成。在此之前，结论只能写成「生成次数相同时」。"]
    lines += ["", "## 训练曲线（每 200 步，第二阶段 20 个，不并入第一阶段）", "",
              "| 组 | 线 | 步 | 验证 loss | 不给历史 vNRR | 给历史 vNRR |", "|---|---|---:|---:|---:|---:|"]
    curves = []
    for line in ("main", "cooldown"):
        for g in "ABC":
            folder = out / line / g
            for probe in sorted(folder.glob("probe_*.json"), key=lambda p: int(p.stem.split("_")[1])):
                pr = json.loads(probe.read_text())
                step = pr["step"]
                val = folder / f"validation_{step}.json"
                loss = json.loads(val.read_text())["loss"] if val.exists() else float("nan")
                v0 = pr["None"]["phase1"]["vNRR"]
                key = [k for k in pr if k not in ("None", "step", "group")]
                v1 = pr[key[0]]["phase1"]["vNRR"] if key else float("nan")
                lines.append(f"| {g} | {'主线' if line == 'main' else '收尾支线'} | {step} | {loss:.4f} | "
                             f"{v0:.1%} | {'' if v1 != v1 else f'{v1:.1%}'} |")
                curves.append([g, line, step, loss, v0, v1])
    lines += ["", "主线学习率未降；收尾支线第 800–1,000 步学习率降到 0。曲线只看趋势，不用来挑模型。", ""]
    # only the real medium run writes the shared report; other configs (smoke) keep theirs local
    if Path(str(cfg["output"])).name == "medium":
        path = ROOT / "experiment-results/history-benefits/中等规模报告.md"
    else:
        path = out / "报告.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}", flush=True)

    run = Run(cfg, out, "report", "report")
    run.table("final/main_table", ["condition", "n", "vNRR", "fNRR", "oracle_val", "phase1_vNRR", "seconds"], table)
    run.table("final/paired", ["comparison", "n", "diff", "ci_low", "ci_high"], comps)
    run.table("curves/probes", ["group", "line", "step", "val_loss", "vNRR_no_history", "vNRR_history"], curves)
    run.summary({f"vNRR/{r[0]}": r[2] for r in table})
    run.finish()

"""Tables of the mix run (正式实验规则.md §12) -> <output>/报告.md, and to W&B.

The written analysis goes to experiment-results/history-benefits/近似历史小规模报告.md (by hand,
from these tables). Medium-run results on the same 200 problems with the same eval seeds are
paired in as references.
"""

from typing import Any
import json
from collections import Counter

import numpy as np

from common import ROOT, train_seed
from medium_report import paired, solved

CONDITION_LABELS = {
    "self_C_two": "C 两阶段（自己先 300，再带历史 700）",
    "self_C_direct": "C 直接生成 1000",
    "self_A_direct": "A 直接生成 1000",
}
MEDIUM_REFS = {
    "self_T8_direct": "原 T8 直接生成 1000（同代码，上一轮）",
    "self_A_direct": "上一轮 A 直接生成 1000",
    "self_C_two": "上一轮 C 两阶段",
}
COMPARISONS = [
    ("self_C_two", "self_C_direct", "精修有没有用：C 两阶段 − C 直接"),
    ("self_C_two", "self_A_direct", "C 方案整体 − A"),
    ("self_C_direct", "self_A_direct", "C 不用历史时 − A（混合训练有没有伤直接生成）"),
    ("self_A_direct", "medium:self_T8_direct", "A − 原 T8"),
    ("self_C_two", "medium:self_T8_direct", "C 两阶段 − 原 T8"),
    ("self_A_direct", "medium:self_A_direct", "这轮 A − 上一轮 A（少重复用题的效果）"),
    ("self_C_two", "medium:self_C_two", "这轮 C 两阶段 − 上一轮 C 两阶段"),
]


def rows_of(out: Any, cfg: Any, name: str) -> Any:
    if name.startswith("medium:"):
        path = ROOT / cfg["reuse"] / "final" / f"{name.split(':', 1)[1]}.json"
    else:
        path = out / "final" / f"{name}.json"
    return json.loads(path.read_text())["rows"] if path.exists() else None


def rate(rows: Any, key: str, part: str = "merged") -> float:
    return sum(bool(r[part] and r[part][key]) for r in rows) / len(rows)


def pct(v: Any) -> str:
    return "—" if v is None or v != v else f"{100 * v:.1f}%"


FVU_BINS = ((0.0, 1e-3, "FVU < 1e-3"), (1e-3, 0.1, "FVU 1e-3–0.1"), (0.1, float("inf"), "FVU ≥ 0.1"))


def fvu_bin(f: Any) -> str:
    value = float("inf") if f is None else float(f)
    return next(name for lo, hi, name in FVU_BINS if lo <= value < hi)


def copy_rate_table(rows: list) -> list:
    """Exploratory (规则 §12.7): copy rate and success by the support FVU of the given history."""
    groups: dict = {}
    for r in rows:
        if r.get("history") is None or not r.get("phase1"):
            continue
        groups.setdefault(fvu_bin(r["history"]["fvu"]), []).append(r)
    if not groups:
        return []
    lines = ["### 照抄率：按给的历史好坏分组（C 两阶段，探索性，规则 §12.7）", "",
             "照抄率 = 第二阶段候选里与历史骨架相同的比例。理想情况：好历史多抄，差历史少抄。", "",
             "| 历史在支撑点上的 FVU | 题数 | 平均照抄率 | 第二阶段解对率（vNRR） | 整题解对率（vNRR） |",
             "|---|---|---|---|---|"]
    for _, _, name in FVU_BINS:
        g = groups.get(name, [])
        if not g:
            continue
        copy = float(np.mean([r["phase1_same_as_history"] / max(r["phase1"]["n_candidates"], 1) for r in g]))
        lines.append(f"| {name} | {len(g)} | {pct(copy)} | {pct(rate(g, 'top1_val', 'phase1'))} | "
                     f"{pct(rate(g, 'top1_val'))} |")
    return lines + [""]


def ablation_tables(out: Any) -> list:
    """Exploratory (规则 §12.7): input ablation of the history, one table per checkpoint."""
    files = sorted((out / "ablation").glob("*.json"))
    if not files:
        return []
    lines = ["## 历史的输入消融（验证集，目标交叉熵损失，探索性，规则 §12.7）", "",
             "同一个样本打五次分，只改历史：不给 / 完整 / 只给公式（预测值清零）/ 预测值错配（换成别的题的）/ "
             "预测值完美（换成 y）。只做前向计算，不训练。", ""]
    for path in files:
        r = json.loads(path.read_text())
        lines += [f"### `{r['checkpoint']}`（跳过 {r['skipped']} 个样本）", "",
                  "| 组 | 样本数 | 不给 | 完整 | 只给公式 | 预测值错配 | 预测值完美 | 完整比不给更低 |",
                  "|---|---|---|---|---|---|---|---|"]
        for label, row in r["table"].items():
            cells = " | ".join(f"{row[c]:.3f}" for c in ("none", "full", "tokens_only", "mismatched", "perfect"))
            lines.append(f"| {label} | {row['n']} | {cells} | {pct(row['helped_vs_none'])} |")
        lines.append("")
    return lines


def report(cfg: Any, out: Any) -> None:
    lines = ["# 近似历史小规模验证：自动表格", "",
             f"配置：`{cfg['output']}/config.yaml`；规则：正式实验规则.md §12。题数小时请看置信区间。", ""]
    summary: dict = {}

    # -- training: selected checkpoint, selection loss per checkpoint ----------------------
    lines += ["## 选中的 checkpoint（预注册规则：验证集损失最低，打平取后面的步）", "",
              "| 组 | 选中步 | 验证集损失 |", "|---|---|---|"]
    for g in ("A", "C"):
        path = out / "train" / g / "selected.json"
        if path.exists():
            sel = json.loads(path.read_text())
            lines.append(f"| {g} | {sel['step']} | {sel['loss']:.4f} |")
            summary[f"selected_step_{g}"] = sel["step"]
    lines += ["", "## 验证集损失（目标公式的交叉熵，按样本类型；验证集 = 题号 200000 起，代码里叫 selection）", "",
              "| 组 | 步 | 不给历史 | 接近正确 | 真实候选 | 加权（挑选标准） |", "|---|---|---|---|---|---|"]
    for g in ("A", "C"):
        files = sorted((out / "train" / g).glob("validation_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        for path in files:
            v = json.loads(path.read_text())
            cell = lambda k: f"{v[k]:.3f}" if k in v else "—"  # noqa: E731
            lines.append(f"| {g} | {v['step']} | {cell('ce_none')} | {cell('ce_near')} | {cell('ce_real')} | {v['loss']:.3f} |")

    # -- training samples: composition and near-correct levels -----------------------------
    lines += ["", "## 训练样本构成（samples.jsonl）", "",
              "| 组 | 类型 | 样本数 | 占比 | 历史进入序列的比例 |", "|---|---|---|---|---|"]
    levels: Counter = Counter()
    edits: Counter = Counter()
    for g in ("A", "C"):
        path = out / "train" / g / "samples.jsonl"
        if not path.exists():
            continue
        kinds: Counter = Counter()
        valid: Counter = Counter()
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                kinds[r["kind"]] += 1
                valid[r["kind"]] += r["actual_k"]
                if g == "C" and r["kind"] == "near":
                    levels[(r["level"], r["actual"])] += 1
                    edits.update(r["edits"])
        total = sum(kinds.values())
        for k, n in sorted(kinds.items()):
            lines.append(f"| {g} | {k} | {n} | {pct(n / total)} | {pct(valid[k] / n)} |")
    if levels:
        n = sum(levels.values())
        lines += ["", "接近正确的历史，实际档位（要求档 → 实际档）：", "",
                  "| 要求 | 实际 | 占比 |", "|---|---|---|"]
        lines += [f"| {a} | {b} | {pct(c / n)} |" for (a, b), c in sorted(levels.items())]
        lines += ["", "结构改动种类：" + "，".join(f"{k} {v}" for k, v in edits.most_common()), ""]

    # -- final evaluation -------------------------------------------------------------------
    lines += ["## 完整流程评测（开发集 = 题号 100000 起的 200 道，代码里叫 validation）", "",
              "| 条件 | 题数 | vNRR（主指标） | fNRR | Oracle | 平均秒/题 |", "|---|---|---|---|---|---|"]
    table = []
    for name, label in [*CONDITION_LABELS.items(), *((f"medium:{k}", v) for k, v in MEDIUM_REFS.items())]:
        rows = rows_of(out, cfg, name)
        if not rows:
            continue
        rec = [label, len(rows), rate(rows, "top1_val"), rate(rows, "top1_fit"), rate(rows, "oracle_val"),
               float(np.mean([r["seconds_total"] for r in rows]))]
        table.append(rec)
        lines.append(f"| {rec[0]} | {rec[1]} | {pct(rec[2])} | {pct(rec[3])} | {pct(rec[4])} | {rec[5]:.1f} |")
        if not name.startswith("medium:"):
            summary[f"{name}/vNRR"] = rec[2]
    lines += ["", "## 配对比较（vNRR 差值，同题配对，bootstrap 10,000 次，95% 置信区间）", "",
              "| 比较 | 题数 | 差值 | 95% 区间 |", "|---|---|---|---|"]
    verdict: dict = {}
    for i, (a, b, label) in enumerate(COMPARISONS):
        ra, rb = rows_of(out, cfg, a), rows_of(out, cfg, b)
        if not ra or not rb:
            continue
        d = paired(solved(ra), solved(rb), train_seed(cfg) + i)
        verdict[(a, b)] = d
        lines.append(f"| {label} | {d['n']} | {100 * d['diff']:+.1f}pp | [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] |")
    c_two = rows_of(out, cfg, "self_C_two")
    if c_two:
        same = [r["phase1_same_as_history"] / max(r["phase1"]["n_candidates"], 1) for r in c_two if r.get("phase1")]
        lines += ["", f"C 两阶段第二阶段里，和历史骨架相同的候选平均占 {pct(float(np.mean(same)))}。", ""]
        lines += copy_rate_table(c_two)

    # -- diagnostics ----------------------------------------------------------------------
    diag_path = out / "diagnostics" / "history_use.json"
    learned = None
    if diag_path.exists():
        diag = json.loads(diag_path.read_text()).get("C", {})
        gen = diag.get("generation", {})
        lossd = diag.get("loss_delta_vs_none", {})
        lines += ["## 读历史诊断（C，选中的 checkpoint，开发集前 100 道）", "",
                  "| 给什么历史 | 题数 | Oracle（候选里有解对的） | Top-1 | 目标 loss 变化（相对不给） |",
                  "|---|---|---|---|---|"]
        for k, v in gen.items():
            dl = lossd.get(k)
            delta = "—" if dl is None else f"{dl['mean']:+.3f}"
            lines.append(f"| {k} | {v['n']} | {pct(v['oracle_val'])} | {pct(v['top1_val'])} | {delta} |")
        if "exact" in gen and "none" in gen:
            learned = gen["exact"]["oracle_val"] >= 0.5 and gen["exact"]["oracle_val"] > gen["none"]["oracle_val"]
            summary["diagnose/exact_oracle"] = gen["exact"]["oracle_val"]
            summary["diagnose/none_oracle"] = gen["none"]["oracle_val"]
        lines.append("")

    lines += ablation_tables(out)

    # -- pre-registered criteria -----------------------------------------------------------
    lines += ["## 预注册的判断标准", ""]
    if learned is not None:
        lines.append(f"- 学会读历史（原样历史的 Oracle ≥ 50% 且高于不给）：**{'是' if learned else '否'}**")
    for key, label in ((("self_C_two", "self_C_direct"), "C 两阶段 > C 直接"),
                       (("self_C_two", "self_A_direct"), "C 两阶段 > A 直接")):
        if key in verdict:
            ok = verdict[key]["lo"] > 0
            lines.append(f"- {label}（95% 区间下限 > 0）：**{'是' if ok else '否'}**")
            summary[f"criterion/{key[0]}>{key[1]}"] = ok
    if learned is not None:
        summary["criterion/learned_to_read"] = learned
    (out / "报告.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    from wb import Run

    run = Run(cfg, out, "report", "report")
    run.summary(summary)
    if table:
        run.table("final/self", ["condition", "n", "vNRR", "fNRR", "oracle_val", "mean_seconds"], table)
    run.finish()

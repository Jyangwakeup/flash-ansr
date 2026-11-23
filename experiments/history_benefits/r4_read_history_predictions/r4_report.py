"""Tables of round 4 (设计与预注册.md §14) -> <output>/报告.md and W&B.

The pre-registered criteria are computed here from the per-sample ablation losses, paired by
problem: per problem the mean over its samples in the group, then core/stats.paired (bootstrap
10,000, 95% interval).
"""

from typing import Any
import json

import numpy as np

from common import dump
from r4_samples import BIN_NAMES
from stats import paired

K3_MARGIN = 0.02
CHECKPOINTS = ("r3_C", "I", "IC")
LABEL = {"r3_C": "第 3 轮 C 组", "I": "I 组（新初始化）", "IC": "IC 组（新初始化 + 按误差均衡）"}


def per_problem(rows: list, cond: str, keep: Any) -> dict:
    by: dict = {}
    for r in rows:
        if keep(r):
            by.setdefault(r["pid"], []).append(r["ce"][cond])
    return {p: float(np.mean(v)) for p, v in by.items()}


def criteria(rows: list, seed: int = 0) -> dict:
    """K1-K4 of 设计与预注册.md §14 on one checkpoint's ablation rows."""
    hist = lambda r: True  # noqa: E731 -- every row of the ablation is a history sample
    low = lambda r: r["bin"] == 0  # noqa: E731
    real = lambda b: (lambda r: r["kind"] == "real" and r["bin"] == b)  # noqa: E731
    k1_all = paired(per_problem(rows, "mismatched", hist), per_problem(rows, "full", hist), seed)
    k1_low = paired(per_problem(rows, "mismatched", low), per_problem(rows, "full", low), seed)
    k2 = paired(per_problem(rows, "none", real(0)), per_problem(rows, "full", real(0)), seed)
    k3 = paired(per_problem(rows, "full", real(2)), per_problem(rows, "none", real(2)), seed)
    gains = {kind: [float(np.mean([r["ce"]["none"] - r["ce"]["full"] for r in rows
                                   if r["kind"] == kind and r["bin"] == b] or [np.nan])) for b in range(3)]
             for kind in ("real", "near")}
    k4 = {kind: bool(g[0] > g[1] > g[2]) for kind, g in gains.items()}
    return dict(
        K1=dict(all=k1_all, low=k1_low, passed=bool(k1_all["lo"] > 0 and k1_low["lo"] > 0)),
        K2=dict(test=k2, passed=bool(k2["lo"] > 0)),
        K3=dict(test=k3, margin=K3_MARGIN, passed=bool(k3["hi"] < K3_MARGIN)),
        K4=dict(gains=gains, passed=bool(k4["real"] and k4["near"]), by_kind=k4),
    )


def fmt_ci(t: dict) -> str:
    return f"{t['diff']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]（{t['n']} 题）"


def report(cfg: Any, out: Any) -> None:
    lines = ["# 第 4 轮 读历史预测值：自动表格", "",
             f"配置：`{cfg['output']}/config.yaml`；规则：experiment-results/history_benefits/"
             "r4_read_history_predictions/设计与预注册.md §14。题数小时请看置信区间。", ""]
    wb_tables: dict = {}

    link = json.loads((out / "link_complete.json").read_text())
    lines += ["## 起点检查（link）", "",
              "| 项 | 值 |", "|---|---|",
              f"| 历史列重新初始化的标准差（= x 列权重的标准差） | {link['init_std']:.5f} |",
              f"| 列范数 x / y / 历史（第 3 轮 step0） | {link['norms_before']['norm_x']:.2f} / "
              f"{link['norms_before']['norm_y']:.2f} / {link['norms_before']['norm_history']:.2f} |",
              f"| 列范数 x / y / 历史（本轮 step0） | {link['norms_after']['norm_x']:.2f} / "
              f"{link['norms_after']['norm_y']:.2f} / {link['norms_after']['norm_history']:.2f} |",
              f"| 其他权重与第 3 轮 step0 相同 | {link['others_identical']} |",
              f"| 不给历史的样本，第 0 步损失最大差 | {link['none_ce_max_abs_diff']:.2e} |",
              f"| 带历史的样本，第 0 步损失平均差（新 − 旧） | {link['history_ce_mean_diff']:+.4f} |", ""]

    # training sample composition
    lines += ["## 训练样本构成（samples.jsonl）", "",
              "| 组 | 类型 | 样本数 | 历史进入序列 | FVU<1e-3 | 1e-3–0.1 | ≥0.1 | 落不进目标档 |", "|---|---|---|---|---|---|---|---|"]
    for g in ("I", "IC"):
        path = out / "train" / g / "samples.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.open()]
        for kind in ("none", "near", "real"):
            rs = [r for r in rows if r["kind"] == kind]
            if not rs:
                continue
            valid = [r for r in rs if r.get("reason") == "included"]
            bins = [r.get("history_bin") for r in rs if r.get("history_bin") is not None]
            share = [f"{100 * bins.count(b) / len(bins):.0f}%" if bins else "—" for b in range(3)]
            fb = [r for r in rs if "fallback" in r]
            miss = f"{100 * sum(r['fallback'] for r in fb) / len(fb):.1f}%" if fb else "—"
            lines.append(f"| {g} | {kind} | {len(rs)} | {100 * len(valid) / len(rs):.1f}% | " + " | ".join(share) + f" | {miss} |")
    lines.append("")

    # evaluation-set losses during training
    lines += ["## 训练中的固定评测集损失（验证集，题号 200000 起；目标交叉熵）", "",
              "| 组 | 步 | 不给 | 近似 " + " | 近似 ".join(BIN_NAMES) + " | 真实 " + " | 真实 ".join(BIN_NAMES)
              + " | 历史列范数 |", "|" + "---|" * 10]
    for g in ("I", "IC"):
        for path in sorted((out / "train" / g).glob("validation_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            v = json.loads(path.read_text())
            cell = lambda k: f"{v[k]:.3f}" if k in v else "—"  # noqa: E731
            lines.append(f"| {g} | {path.stem.split('_')[1]} | {cell('ce_none')} | "
                         + " | ".join(cell(f"ce_near_b{b}") for b in range(3)) + " | "
                         + " | ".join(cell(f"ce_real_b{b}") for b in range(3)) + f" | {v.get('norm_history', float('nan')):.2f} |")
    lines.append("")

    # ablation + criteria
    results: dict = {}
    for name in CHECKPOINTS:
        path = out / "ablation" / f"{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text())
    if results:
        lines += ["## 历史的输入消融（验证集，目标交叉熵，按 FVU 档分组）", "",
                  "五个条件：不给 / 完整 / 只给公式（预测值清零）/ 预测值错配（换成另一道题的）/ 预测值完美（换成 y）。", ""]
        for name, res in results.items():
            rows = res["rows"]
            lines += [f"### {LABEL[name]}：`{res['checkpoint']}`（丢弃 {res['dropped']} 个样本）", "",
                      "| 组 | 样本数 | 不给 | 完整 | 只给公式 | 预测值错配 | 预测值完美 | 完整比不给更低 |", "|---|---|---|---|---|---|---|---|"]
            table = []
            for kind in ("near", "real"):
                for b in range(3):
                    rs = [r for r in rows if r["kind"] == kind and r["bin"] == b]
                    if not rs:
                        continue
                    m = {c: float(np.mean([r["ce"][c] for r in rs])) for c in res["conditions"]}
                    helped = float(np.mean([r["ce"]["full"] < r["ce"]["none"] for r in rs]))
                    lines.append(f"| {kind} {BIN_NAMES[b]} | {len(rs)} | " + " | ".join(f"{m[c]:.3f}" for c in res["conditions"])
                                 + f" | {100 * helped:.1f}% |")
                    table.append([f"{kind} {BIN_NAMES[b]}", len(rs), *(m[c] for c in res["conditions"]), helped])
            wb_tables[f"ablation/{name}"] = (["group", "n", *res["conditions"], "full<none"], table)
            lines.append("")
        lines += ["## 预注册的判断标准（设计与预注册.md §14）", "",
                  "| 标准 | " + " | ".join(LABEL[n] for n in results) + " |", "|---|" + "---|" * len(results)]
        crit = {n: criteria(r["rows"]) for n, r in results.items()}
        lines.append("| K1 预测值错配 − 完整（全部历史样本），下限 > 0 | " + " | ".join(fmt_ci(c["K1"]["all"]) for c in crit.values()) + " |")
        lines.append("| K1 预测值错配 − 完整（FVU<1e-3），下限 > 0 | " + " | ".join(fmt_ci(c["K1"]["low"]) for c in crit.values()) + " |")
        lines.append("| K2 真实 FVU<1e-3：不给 − 完整，下限 > 0 | " + " | ".join(fmt_ci(c["K2"]["test"]) for c in crit.values()) + " |")
        lines.append(f"| K3 真实 FVU≥0.1：完整 − 不给，上限 < {K3_MARGIN} | " + " | ".join(fmt_ci(c["K3"]["test"]) for c in crit.values()) + " |")
        lines.append("| K4 不给 − 完整 随 FVU 档递减（真实；近似） | " + " | ".join(
            "真实 " + " > ".join(f"{x:.3f}" for x in c["K4"]["gains"]["real"]) + "；近似 "
            + " > ".join(f"{x:.3f}" for x in c["K4"]["gains"]["near"]) for c in crit.values()) + " |")
        for k in ("K1", "K2", "K3", "K4"):
            lines.append(f"| **{k} 达到** | " + " | ".join("**是**" if c[k]["passed"] else "否" for c in crit.values()) + " |")
        lines.append("")
        dump(out / "criteria.json", crit)

    probes = {n: json.loads((out / "copy_probe" / f"{n}.json").read_text())
              for n in CHECKPOINTS if (out / "copy_probe" / f"{n}.json").exists()}
    if probes:
        lines += ["## 照抄检查（描述性，不参与判断）：给原样答案，teacher forcing", "",
                  "| checkpoint | 历史 | 样本数 | 结构符号猜对 | 整条结构全对 | 常数字节猜对 |", "|---|---|---|---|---|---|"]
        for n, p in probes.items():
            for k in ("none", "exact"):
                s = p["result"][k]
                lines.append(f"| {LABEL[n]} | {'不给' if k == 'none' else '原样答案'} | {s['samples']} | "
                             f"{100 * s['structure_acc']:.1f}% | {s['whole_structure_right']} / {s['samples']} | {100 * s['byte_acc']:.1f}% |")
        lines.append("")

    (out / "报告.md").write_text("\n".join(lines))
    print("\n".join(lines))

    from wb import Run

    run = Run(cfg, out, "report", "report")
    for key, (columns, rows) in wb_tables.items():
        run.table(key, columns, rows)
    if results:
        run.summary({f"criteria/{n}/{k}": c[k]["passed"] for n, c in crit.items() for k in c})
    run.finish()

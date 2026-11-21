"""Render an engineering status report from measured artifacts only."""

from typing import Any
import json
import re
from datetime import datetime, timezone
from statistics import mean, median
from common import ROOT


def report(cfg: Any, out: Any, error: Any = None) -> Any:
    def read(path: Any) -> Any:
        return json.loads(path.read_text()) if path.exists() else None

    eval_done = all(
        (out / f"pilot/{g}/evaluation_complete.json").exists() for g in "ABC"
    )
    storage_done = "regenerate" in (read(out / "storage/costs.json") or {})
    status = "试跑尚在进行，完成范围见文末。"
    if eval_done:
        status = "200 题三组短训与末步评测已完成，历史训练、生成和拟合链路已跑通。"
        status += "存储对照也已完成。" if storage_done else "存储全量重建测量仍在进行。"
    lines = [
        "# 历史精修工程试跑报告",
        "",
        status,
        "",
        "这是工程正确性与 CPU 成本检查，不用于判断历史精修是否优于独立采样。",
        f'配置：`experimental/history_benefits/pilot.yaml`；原始产物：`{cfg["output"]}/`。',
        f'报告更新时间：{datetime.now(timezone.utc).isoformat(timespec="seconds")}。',
        "",
    ]
    if eval_done and storage_done:
        lines += """
## 结论与下一步

**本轮工程试跑通过，可以进入效果验证的准备阶段；目前还不能得出“历史精修比无历史生成更好”的结论。**

- 已验证：历史公式前缀和历史预测列能进入模型，参与训练并影响输出；三组从共同起点完成 100 次更新、恢复检查和末步评测，B/C 实际有效历史约 43.4%。
- 已测得：保存候选池比重新生成拟合便宜得多；完整方式约 23.65 MiB、整池回放约 0.30 秒，精简方式约 2.46 MiB、含重算预测约 2.03 秒，全量重建约 28.2 分钟且 400/400 池一致。这是候选池复用收益，不是历史模型质量收益。
- 已接受的限制：编号 40 的题在当前拟合流程下没有可用候选。沿用原处理，保留失败记录，训练回退 K=0；不把数值修复设为继续实验的前提。
- 尚未证明：历史是否提高正确率或降低预测误差、是否在相同推理成本下更有效，以及同结构缓存能否在真实任务中加速。真实缓存命中为 0，不能以构造测试替代收益证据。
- 需要单独核对：T8 权重开启 T8.1 修复后的能力变化。共同起点使 A/B/C 可比较，但不保证原能力没有损失，也不保证 100 步已充分适应。

建议下一步按以下顺序推进（这是建议，不表示已经启动新训练）：

1. **先分析现有结果并补齐迁移对照。** 在相同验证题、生成预算、拟合和评分设置下，比较冻结原 T8、修复后的共同 step-0、A 的末步模型；已有产物能复用的先复用，缺失的条件再补测。分别统计求解失败、数值误差和统一成功标准下的成功率，判断修复适应是否是主要干扰因素。同时配对比较 B/C 的 K=0 与 K=1，作为诊断，不据此宣称正式收益。
2. **固定正式评测口径与运行配置。** 保留 A 无历史微调对照；第 40 题类失败沿用原规则，所有题保留在统计分母中。训练完整保存候选池，便于复查；缓存结果独立记录。确认同 draws 与等推理时间两种比较、种子、数据划分和 checkpoint 选择规则，避免事后选择有利口径。
3. **再按设计扩到正式规模。** 目标仍为 10,000 训练题、1,000 验证题、A/B/C 各 10,000 次更新、第一版 K=1。已有 CPU 测量估计建池约 30.9 小时、三组训练约 51.2 小时，两者之外还有正式评测等开销；先根据可用硬件落实运行预算。此次报告更新不自动启动该实验。

最接近的下一项工作是“现有结果分析＋迁移对照补测”，而不是再重复一轮相同工程试跑，也不是先为单道极端题改写整个拟合流程。

""".splitlines()
    if error:
        lines += [
            "## 当前停止原因",
            "",
            str(error),
            "",
            "未通过前置检查时，不自动扩大到 200 题训练。",
            "",
        ]
    checks = read(out / "acceptance.json")
    if checks:
        lines += [
            "## 历史输入检查",
            "",
            f'有效 K=1 检查样本数：{checks["valid_history_samples"]}。',
            "已验证因果遮罩、历史标签忽略、缺失历史数值不影响输出、历史梯度和数值输入影响。",
            f'保存预测重新求值一致：{checks.get("stored_predictions_match", False)}。',
            "",
        ]
    migration = read(out / "migration.json")
    if migration:
        lines += [
            "## 权重迁移",
            "",
            f'T8 → T8.1 修复后的旧词表 logits 最大差异：{migration["old_to_fixed_max_abs"]:.6g}。',
            f'T8.1 → 无历史扩展模型的旧词表 logits 最大差异：{migration["fixed_to_extended_max_abs"]:.6g}。',
            "这项检查带有实际使用的 NaN 数值通道。T8.1 移除了 T8 预训练时依赖的非数值位置嵌入，因此它会改变生成行为；这不是历史信息带来的效果。",
            "新增输出行采用预训练输出行均值初始化，输入嵌入按固定种子初始化。最初随机输出行引起异常标记生成的诊断保存在 diagnostics/ 中，不作为正式结果。",
            "",
        ]
    detailed = read(out / "migration_detailed.json")
    if detailed:
        lines += [
            "在带填充的双题探针中，单独开启 query norm 修复的最大 logits 变化为 "
            f'{detailed["max_abs_change_from_T8"]["query=True,numeric=False"]:.3f}，'
            "单独开启数值通道修复为 "
            f'{detailed["max_abs_change_from_T8"]["query=False,numeric=True"]:.3f}。'
            "这只是该探针的测量，不代表所有输入的变化幅度。",
            "",
        ]
    pool_seconds = 0.0
    lines += [
        "## 候选池与失败记录",
        "",
        "| 来源 | 题数 | 保留候选数 | 空池题号 | 累计生成拟合秒数 |",
        "|---|---:|---:|---|---:|",
    ]
    for kind in ("random", "model"):
        pools = [read(p) for p in (out / "pools" / kind).glob("*.json")]
        seconds = sum(p["seconds"] for p in pools)
        pool_seconds += seconds
        empty = sorted(p["problem_id"] for p in pools if not p["candidates"])
        lines.append(
            f'| {kind} | {len(pools)} | {sum(len(p["candidates"]) for p in pools)} | {empty} | {seconds:.1f} |'
        )
    numeric = read(out / "numeric_range_diagnostics.json") or []
    if numeric:
        lines += ["", "数值范围诊断：", ""]
        for row in numeric:
            lines.append(
                f'- 第 {row["pid"]} 题观测值仍有限，但最大绝对值约 {row["max_abs"]:.3e}，方差计算溢出。题目没有删除，历史为空时回退 K=0。'
            )
    lines += [
        "",
        "原始模型池沿用生成器默认总长度 64；第二阶段按历史前缀加 128 个目标 token 检查 256 上限。详细解析参数见 `resolved_generation.json`。",
        "",
    ]
    lines += [
        "## 实际训练与生成",
        "",
        "| 阶段/组 | 更新数 | 耗时/秒 | 有效历史比例 | 恢复检查 |",
        "|---|---:|---:|---:|---|",
    ]
    training_seconds = 0.0
    for phase in ("small", "pilot"):
        for group in ("A", "B", "C"):
            s = read(out / f"{phase}/{group}/summary.json")
            if s:
                lines.append(
                    f'| {phase}/{group} | {s["steps"]} | {s["seconds"]:.1f} | {s["history_fraction"]:.1%} | {s["resume_checked"]} |'
                )
                if phase == "pilot":
                    training_seconds += s["seconds"]
    lines += [
        "",
        "B/C 的预定无历史、条件 dropout 和空池回退分别统计；预定 50% 无历史不等于最终恰好 50% 有历史。",
        "",
    ]
    for group in ("B", "C"):
        s = read(out / f"pilot/{group}/summary.json")
        if s:
            lines.append(
                f'- {group}：预定 K=1 共 {s["planned_history"]} 次，实际 {s["actual_history"]} 次；原因计数 `{s["reasons"]}`。'
            )
    lines += ["", "每组实际更新耗时（不含更新之间的验证和恢复探针）：", ""]
    for group in ("A", "B", "C"):
        path = out / f"pilot/{group}/metrics.jsonl"
        if path.exists():
            timings = [
                json.loads(line)["seconds"] for line in path.read_text().splitlines()
            ]
            if timings:
                lines.append(
                    f"- {group}：平均 {mean(timings):.3f} 秒，中位数 {median(timings):.3f} 秒。"
                )
    lines += [
        "",
        "| 阶段/组 | K | 题数 | 拟合后候选总数 | 提取失败数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase in ("small", "pilot"):
        for group in ("A", "B", "C"):
            ev = read(out / f"{phase}/{group}/evaluation.json") or []
            for k in (0, 1):
                rows = [r for r in ev if r["requested_history"] == k]
                if rows:
                    lines.append(
                        f'| {phase}/{group} | {k} | {len(rows)} | {sum(len(r["candidates"]) for r in rows)} | {sum(len(r.get("extraction_failures", [])) for r in rows)} |'
                    )
    lines += [
        "",
        "拟合后的候选包含原有拟合器生成的常数变体，因此候选总数可以超过原始 draws，不能用两者相除计算生成成功率。每个评测条件实际原始生成均为 20 次。",
        "",
    ]
    total_hits = 0
    for group in ("A", "B", "C"):
        rows = read(out / f"pilot/{group}/evaluation.json") or []
        total_hits += sum(r.get("cache_replay", {}).get("hits", 0) for r in rows)
    lines.append(
        f"真实评测中的同结构缓存命中共 {total_hits} 次。构造的同公式重复项检查见 `cache_hit_check.json`；它是功能检查，不是模型带来的缓存收益。"
    )
    costs = read(out / "storage/costs.json")
    if costs:
        lines += [
            "",
            "## 候选池保存成本",
            "",
            "以下都覆盖同一批 200 题的两种来源。完整和精简方式使用相同候选、相同顺序与预加载的 X。精简文件仅保留 expression_prefix 和 constants 两个字段。",
            "",
            "| 方式 | 文件 MiB | 写入秒数 | 回放 1 次 | 回放 2 次 | 回放 10 次 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        storage_details = []
        for name in ("full", "formula_constants"):
            if name not in costs:
                continue
            v = costs[name]
            ts = {r["repetitions"]: r["total_seconds"] for r in v["reads"]}
            lines.append(
                f'| {name} | {v["bytes"] / 2**20:.2f} | {v["write_seconds"]:.3f} | {ts[1]:.3f}s | {ts[2]:.3f}s | {ts[10]:.3f}s |'
            )
            first = v["reads"][0]
            storage_details.append(
                f'{name} 首次回放：文件读取 {first["read_seconds"]:.3f}s，反序列化 {first["deserialize_seconds"]:.3f}s，预测重算 {first["recompute_seconds"]:.3f}s；tracemalloc 可见分配峰值 {first["python_peak_bytes"] / 2**20:.2f} MiB，进程累计 RSS 高水位 {first["process_lifetime_peak_rss"] / 2**30:.2f} GiB。\n'
            )
        lines.extend(["", *storage_details, ""])
        if "regenerate" in costs:
            v = costs["regenerate"]
            lines += [
                f'全量重新生成和拟合实际耗时 {v["measured_rebuild_seconds"]:.1f} 秒，复现一致的池为 {v["identical_pools"]}/{v["total_pools"]}。不一致清单保存在 `storage/costs.json`。',
                "",
            ]
            lines.append(
                f'不保存时，回放 2、10 次的线性成本估计为 {v["measured_rebuild_seconds"] * 2:.1f}、{v["measured_rebuild_seconds"] * 10:.1f} 秒；这两项没有再全量实跑。'
            )
            lines.append("")
            for name in ("full", "formula_constants"):
                value = costs[name]["estimated_break_even_full_pool_replays"]
                if value is not None:
                    lines.append(
                        f"- {name}：写入成本除以每次整池复用节省时间，阈值约为 {value:.6f} 次整池回放；若小于 1，表示第一次复用就已覆盖写入开销。"
                    )
        lines += [
            "",
            "首次读取不是操作系统冷缓存测试。总回放耗时还包括一致性断言；编译后的公式函数在进程内缓存。tracemalloc 不保证覆盖所有原生分配；RSS 是进程累计高水位；精简格式补测使用了新进程，与完整格式的 RSS 不可直接比较，不能据此宣称某种策略的独立峰值内存更低。",
            "训练时一次载入候选后在内存中选用，候选使用次数不等于磁盘读取次数；这里的 1/2/10 次指整池回放，而不是训练步数。没有额外训练三套存储版本。",
            "",
        ]
    lines += ["## 实际复用与正式规模估算", ""]
    for group, kind in (("B", "random"), ("C", "model")):
        s = read(out / f"pilot/{group}/summary.json")
        if s:
            available = sum(
                len(read(out / f"pools/{kind}/{pid}.json")["candidates"])
                for pid in range(cfg["train_problems"])
            )
            used = len(s["usage"])
            lines.append(
                f'- {group}：历史实际使用 {sum(s["usage"].values())} 次，涉及 {used}/{available} 个训练池候选；训练池未使用比例 {1 - used / available:.1%}。保留 16 个但只从前 10 个选择，本来就会留下未使用候选。'
            )
    lines += [
        "",
        f"按首次两类建池累计计时线性外推，10,000 题约需 {pool_seconds * 50 / 3600:.1f} 小时；若另加 1,000 验证题，约 {pool_seconds * 55 / 3600:.1f} 小时。这个估计没有单独补入题目生成和完整正式评测。",
    ]
    if training_seconds:
        lines.append(
            f"三组各 10,000 次更新，按本轮三组短训总耗时线性外推约 {training_seconds * 100 / 3600:.1f} 小时；正式验证规模、验证频率和硬件变化会改变实际成本。"
        )
    if costs and "full" in costs and "formula_constants" in costs:
        lines.append(
            f'若扩到 11,000 题并保持每题两类池最多 16 个，候选文件量线性估算：完整方式 {costs["full"]["bytes"] * 55 / 2**30:.2f} GiB，精简方式 {costs["formula_constants"]["bytes"] * 55 / 2**30:.2f} GiB；不含题目、checkpoint 和评测产物。'
        )
    lines += """
## 问题汇总：已处理、未解决与结论边界

本节区分“已经修复并验证”“已作回退但根因未解决”和“尚未证明的效果”。工程检查通过，不等于模型精修效果已经得到证明。

### 已处理的工程问题

| 问题 | 影响 | 本轮处理与验证 | 当前边界 |
|---|---|---|---|
| 新增历史 token 的输出权重初始化不合适 | 最初随机初始化新增输出行后，模型容易生成异常标记，影响正常公式生成 | 新增输出行改为原输出行均值，输入嵌入仍按固定种子初始化；重新跑小试；旧结果归档到 `diagnostics/random_output_init/` | 属于初始化修正，不是历史精修收益；最终结果使用修正后的共同起点 |
| “没有历史”与“有效历史预测为零”容易混淆 | 仅把缺失预测填为 0，会让模型误读为有效零预测 | 增加独立有效标记，在数值编码后清零缺失槽位；验证改变缺失槽位数字不影响输出，有效历史可影响输出并传递梯度 | K=0 表示本次不提供历史，不等于题目的候选池一定为空 |
| 相同公式的不同表示导致缓存匹配失败 | 公式化简、规范化后的表示不一致，可能漏掉重复项 | 统一公式表示并按可拟合常数位置生成匹配键，保留结构性常数；构造重复公式检查通过 | 仅证明缓存功能可用；真实评测缓存命中仍为 0 |
| 精简保存格式最初包含额外元数据 | 不符合“只保存公式与常数”的比较要求 | 改为仅保存 `expression_prefix` 和 `constants`，保持候选顺序，单独补测；旧格式与测量归档到 `diagnostics/storage_with_metadata/` | 新进程补测的 RSS 不能和旧进程累计峰值直接比较 |
| 存储比较的计时口径需要一致 | 输入加载、公式编译缓存、校验等开销会影响比较 | 两种方式共用预加载 X，明确列出读文件、反序列化、预测重算与总回放时间；披露进程内编译缓存与一致性校验开销 | 首次读取不代表操作系统冷缓存；未得到严格隔离的每种策略峰值内存 |

另有一组容易出错的边界，本轮已实现并检查，但不应说成它们都曾实际发生故障：历史与填充不参与目标损失、目标移位与因果遮罩、提取目标而非历史、删除查询点时同步删除历史预测行、条件 dropout 同时移除两类历史条件，以及保存恢复训练状态。有效 K=1 已实际覆盖，不能把纯 K=0 跑通当作历史链路通过。

### 第 40 题：已经定位数值失败路径，尚未修复根因

这里的“第 40 题”指 `problem_id=40`。题目本身保存在 `runs/history_benefits/pilot/problems/40.json` 和 `40.npz`；两个空池记录在 `pools/model/40.json`、`pools/random/40.json`。题目未删除、未替换，训练选到空池时回退 K=0。

它的正确公式为：

```text
y = x3 + x11 * (sinh(-x15 * (x15 + x7 - x11)) - acosh(x3)) / (x15 * x7) - x1 - x7
```

在保存数组的第 296 行（从 0 开始计数），`sinh` 的输入约为 660.806，正确公式输出约为 `6.869072019030825e285`，带噪声观测值为 `6.867480576834517e285`。全部 460 个观测值仍是有限数，但它们的方差计算已经溢出为 `inf`。例如候选 `y=0` 的均方误差为 `inf`，再除以方差得到 `NaN`。

进一步直接调用本地原有 `Refiner`，在该题带噪声数据上分别检查简单公式 `x1` 和题目的正确公式：二者均得到 `loss=inf`、`valid_fit=False`。正确公式也失败，是因为与带噪声观测之间仍有残差，而残差平方依然可能溢出；这不能解释成正确公式本身错误。

原有流程的具体处理是：

1. 排除 X 或 y 已经非有限的数据行；本题所有观测本身有限，因此不能由这个检查挡住。
2. 常数拟合后计算 `mean((prediction - y) ** 2)`；无可拟合常数的公式也要经过这个误差检查。
3. 损失非有限时，候选不被认作有效拟合；失败候选不进入最终结果。
4. 全部候选失败时，`fit(..., on_empty="return")` 返回空列表；`on_empty="raise"` 则抛出 `ConvergenceError`。本轮使用前者。

因此，问题并不取决于计算在模型内部还是外部：本地原有推理的拟合流程同样会遇到它。历史方案只是把空结果作为无历史回退处理，没有自动修复数值计算。现有建池状态统计区分无效表达式和拟合失败，但不足以还原每个原始候选的具体失败原因；上面的直接检查证明了这条失败路径，不能推断所有候选都只因这一原因失败。这里描述的是当前本地实现，不代表已经核实作者论文实验时的全部处理方式。

当前已确定沿用原拟合流程：失败候选不保留，全部失败返回空池；训练保留该题和已知正确答案，回退 K=0，不强行补候选、不静默换题。评测中无可用答案应计为求解失败，不因回退而算作成功。数值缩放或稳定误差计算可作为后续独立改进，不是本轮继续实验的前置条件；如果将来修改，需对所有对照使用相同规则，并同时验证拟合、FVU 和 MDL 排序。

### T8 权重与 T8.1 修复：迁移可以运行，行为影响仍需评估

本轮加载的是 T8 的预训练权重，没有加载另外一套 T8.1 预训练权重；“启用 T8.1”指开启相应计算修复。权重相同，不代表计算代码改变后输出仍相同。

实际例子：`sin` 这样的非数字 token，其额外数字输入用 NaN 表示“没有数字”。旧行为会把这个 NaN 编码成额外特征；修复后将没有数字的位置的数值嵌入清零。T8 在旧行为下训练，改变这一输入特征会改变下一个 token 的打分。此外还有编码器 query norm 的掩码修复。

迁移探针已测到 T8 与修复后模型的 logits 差异；修复后模型再增加历史输入，在无历史条件下旧词表 logits 保持一致。后者说明历史扩展的迁移检查通过，并不能证明前一步修复不损害原模型能力；新增词表项也会改变 softmax 的归一化范围。

本轮 A/B/C 使用相同修复后的 step-0 起点。A 是同预算的无历史微调对照，用于区分继续训练与引入历史的影响，不能直接由以前未微调的 baseline 替代。但 100 次更新是否足以适应修复、最终能力恢复多少，仍未得到充分验证。正式实验前应单独检查原 T8、修复后起点及微调后的无历史表现；不能把适应修复的变化算作历史收益。

### 尚未得到证明的效果与后续决策

- **历史是否提高公式质量：未证明。** 本轮 160 训练题、40 验证题、每组 100 次更新用于工程检查；下一阶段需要同预算的质量比较，不能把生成数量当作质量，也不能据此调整原答案写法的采样比例。
- **缓存是否实际加速：未证明。** 构造重复项检查成功，真实评测命中为 0。缓存是在同题、同设置下复用旧拟合结果，不保证重新随机拟合绝不会得到更好结果；应继续分开记录原始拟合与缓存复用结果。
- **候选池持久保存是否有价值：本轮支持。** 整池读取或重算预测远快于重新生成拟合，且 400/400 个池全量重建一致；这与同结构缓存命中是两件事。
- **保存格式的正式选择：待确定。** 完整方式便于直接核对预测，精简方式节约空间但需重算；训练目前一次载入后在内存中选用，不能把样本使用次数当成磁盘读取次数。
- **正式规模成本：仅为估算。** 线性外推没有覆盖更大规模评测、硬件变化和所有额外开销；本轮未启动正式万题实验。

结论：历史输入、监督、训练恢复、生成、拟合和保存链路已跑通；数值极端题仅有回退处理，迁移效果仍需独立评估，历史带来的质量收益与真实缓存收益均未被本轮证明。

""".splitlines()
    lines += ["## 检查记录", ""]
    for filename in (
        "tests.log",
        "recipe_tests.log",
        "history_tests.log",
        "post_format_tests.log",
    ):
        path = out / filename
        if path.exists():
            counts = re.findall(r"(\d+) passed", path.read_text())
            if counts:
                lines.append(f"- `{filename}`：{counts[-1]} 项通过。")
    comparison = read(out / "type_comparison.json")
    if comparison:
        lines.append(
            f'- 核心文件类型检查：原版本 {comparison["baseline_errors"]} 项诊断，当前 {comparison["current_errors"]} 项，新增诊断 `{comparison["new_diagnostics"]}`。'
        )
    for filename in ("lint.log", "experiment_types.log"):
        if (out / filename).exists():
            lines.append(f"- 实验脚本静态检查日志：`{filename}`。")
    lines += [
        "",
        "## 完成范围与限制",
        "",
        f'200 题训练完成组数：{sum((out / f"pilot/{g}/summary.json").exists() for g in ("A", "B", "C"))}/3。',
        f'候选池文件数：{len(list((out / "pools").glob("*/*.json")))}，目标为 400（200 题 × 两种来源）。',
        f'存储比较全量重建完成：{bool(costs and "regenerate" in costs)}。',
        "生成失败和回退保留在逐题结果中；训练样本的预定 K、实际 K、无条件状态和原因保存在 samples.jsonl。",
        "最终工程评测使用末步权重；没有根据验证结果挑选最佳 checkpoint。没有执行正式 10,000 题实验。",
        "硬件和依赖见 `environment.json`；时间测量受本机并行任务影响，日志中注明了测试重叠的阶段。",
        "训练调度、配置混用拒绝和验证条件核对分别见 `training_audit.json`、`manifest_guard_check.json`、`evaluation_audit.json`。运行期间仅修复和整理代码，具体版本及语义核对记录在各 provenance 文件和 `code_before_format/` 中。",
        "",
    ]
    target = ROOT / "experiment-results/history-benefits/试跑报告.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines))
    return target

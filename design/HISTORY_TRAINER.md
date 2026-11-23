# 统一训练器（HistoryTrainer）：什么时候做、到时候怎么做

状态：**暂不实施**（2026-11-23 评估）。现在 `experiments/history_benefits/core/train.py` 的"固定循环 + 可换插口"已经够用。第 3、4 轮都只写了自己的 `batch_maker` / `validator`，core 一行没改。

## 为什么现在不做

| 看什么 | 2026-11-23 的情况 |
|---|---|
| 新的一轮接得进来吗 | 能，r3、r4 都没改 core |
| 训练器是不是瓶颈 | 不是。3M 参数的模型，CPU 上 1000–2000 步，几小时跑完 |
| 输入结构定了吗 | 没定。可能变成多条历史、只要公式不要预测、历史改放到别处、完全不同的额外输入。现在包装，接口只能靠猜 |
| 继承原作者的 `Trainer` 能白拿功能吗 | 不能。它调用模型时不传 `history_mask`，数据流水线（`streaming.py`）不生成历史样本，检查点不存随机数状态，取数据、调用模型、`wandb.init` 都写死在长函数中间 |

## 什么时候必须做（满足任意一条就开工）

1. **第一轮要上 GPU 时。** `core/train.py` 的设备写死为 CPU，没有混合精度；样本在主进程里一条一条造，上了 GPU 后造数据会成为瓶颈。
2. **第一轮需要改 `collate` 时**，也就是把样本拼成模型输入的方式变了，比如多条历史，或者历史不再是数据第 19 列。`core/train.py` 直接 import `common.collate`，而 `collate` 写死了 19 列。
3. 模型明显变大，或者单轮训练超过一天。

## 到时候怎么做

**位置。** 新文件 `src/flash_ansr/train/history_trainer.py`。原作者的 `Trainer` 和 `experiments/history_benefits/core/` 都不改，前几轮继续用旧代码，保证可复现。

**和输入结构解耦。** 训练器只认一个约定：

```python
batch = input_format.collate(samples, device)
# batch = {
#   "inputs": {...},          # 原样传给 model(**inputs)，训练器不看里面有什么
#   "labels": LongTensor,     # [b, T-1]，不计 loss 的位置填 <pad>
#   "outlier_target": Tensor | None,
#   "point_mask": BoolTensor | None,
#   "kinds": list[str],       # 每条样本的类别，用于按类别统计 CE
# }
logits = model(**batch["inputs"])[:, :-1]
```

换输入结构时，只需要写一个新的 `collate` 和对应的模型配置开关，训练器不用改。数据列数从 `model.encoder_max_n_variables + model.history_slots` 读，不写死。

**样本来源。** `source.batch(step, rng) -> list[sample]`，每条样本带 `kind`，签名和现在的 `batch_maker` 一样，`MixSampler` 可以直接接上。随机数分两种模式：

- `sample_rng: sequential`：一条随机数流，存进检查点。这是现在的行为。
- `sample_rng: per_step`：每一步的随机数只由 `(seed, step)` 决定。这样可以用多个进程提前造后面几步的数据，进程数不同，结果也一样。

**复用现成代码。**

- `build_optimizer`：`src/flash_ansr/train/optimizers.py`
- `pw_linear_schedule`：`src/flash_ansr/train/schedules.py`
- `_z_loss`：`src/flash_ansr/train/train.py`
- 检查点和随机数状态的保存/恢复：`core/train.py` 的 `checkpoint` / `restore`，加上 GradScaler 和 CUDA 的随机数状态

**新增配置项。** `device`、`amp_dtype`（none / bf16 / fp16，fp16 时开 GradScaler）、`compile`、`grad_clip`（默认 2.0）、loss 权重（默认 0.05 / 1e-4）。W&B 通过传入的 logger 对象记录（有 `.log(dict, step)` 就行），`src/` 不 import `experiments/`。本地照样写 `metrics.jsonl` / `samples.jsonl`。

**验收。**

1. 逐步对数：用 r3 的 `smoke.yaml`、C 组，在 CPU、fp32 下，旧的 `core/train.py` 和新训练器各跑 20 步，每一步的 loss、学习率和最终权重都要完全一致（`atol=0`）。
2. 单元测试 `tests/test_train/test_history_trainer.py`：
   - 小模型（`history_slots=1`）跑 2 步。
   - 一口气跑 4 步，和跑 2 步、存档、恢复后再跑 2 步，权重要完全一致。
   - `per_step` 模式下，进程数不同，样本完全一样。
   - 换一个不带历史的 `collate`，训练器照常运行。
3. 有 GPU 后：用 bf16 跑 50 步，loss 曲线和 CPU fp32 接近。

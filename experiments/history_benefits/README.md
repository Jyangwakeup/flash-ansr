# history_benefits：代码结构

几轮共用的主代码放在 `core/`，每一轮的代码各自一个文件夹。设计、预注册和报告在 `experiment-results/history_benefits/`：每轮一个同名目录，总目录是那里的 `README.md`。

```
core/                           共用主代码，尽量不改
  common.py                     种子、题号、读写题目和候选池、拼训练样本、loss
  pool.py                       出题、建候选池
  train.py                      训练循环（batch_maker / validator 两个接口可换样本来源）
  evaluation.py                 两阶段生成和评分（拟合点 fNRR、留出点 vNRR）
  run_support.py                provenance 记录、加载 step-0 和 checkpoint、jsonl 日志
  stats.py                      配对 bootstrap 比较
  diagnose.py                   诊断：模型有没有在用历史（任何一轮的配置都能用）
  wb.py                         W&B 记录（分组取配置里的 wandb_group）
r1_pipeline_check/              第 1 轮：流程检查
r2_random_vs_model_history/     第 2 轮：A 不给 / B 随机 / C 模型历史
r3_near_correct_history/        第 3 轮：近似正确历史
```

每一轮的文件夹里有：`config.yaml`（正式配置）、`smoke.yaml`（几分钟跑完的冒烟配置）、`r<N>_run.py`（各阶段入口）、`r<N>_report.py`（生成表格），以及 `r<N>_run.sh`（用 tmux 跑完整流程）。

## 怎么跑

```bash
# 第 3 轮完整流程（tmux 会话 r3）
experiments/history_benefits/r3_near_correct_history/r3_run.sh start
# 冒烟
CONFIG=experiments/history_benefits/r3_near_correct_history/smoke.yaml SESSION=r3smoke \
  experiments/history_benefits/r3_near_correct_history/r3_run.sh start
```

结果写到配置里 `output:` 指定的 `runs/history_benefits/<轮次名>/`。

## 新开一轮

1. 新建文件夹 `r<N>_<做什么>/`，名字要让人一眼看出这一轮在比什么，例如 `r4_two_history_slots`。
2. 配置里写 `output: runs/history_benefits/r<N>_<做什么>` 和 `wandb_group: r<N>_<做什么>`。设计与预注册写在 `experiment-results/history_benefits/r<N>_<做什么>/设计与预注册.md`（开跑前锁定），报告写在同一目录的 `报告.md`，这一轮结束时打标签 `hb/r<N>-<做什么>`。
3. 每一轮只 import `core/` 和自己文件夹里的模块，**不 import 别的轮次**。入口脚本开头加一行：
   `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))`
4. 文件名加 `r<N>_` 前缀（`r4_run.py`、`r4_report.py`），这样 W&B 的 job_type 和 mypy 都能分清是哪一轮。

改动放在哪一层，看改的是什么：

| 想改什么 | 例子 | 放在哪 | core 要不要动 |
|---|---|---|---|
| 数字和比例 | 步数、题数、三类样本的比例、学习率 | 只改本轮的 `config.yaml` | 不动 |
| 训练样本怎么造 | 第 3 轮的近似正确历史（`r3_samples.py` + `corrupt.py`） | 本轮文件夹里写一个样本生成器，通过 `train(..., batch_maker=…, validator=…)` 接进去 | 不动 |
| 怎么评测、出什么报告 | 第 3 轮的消融、照抄率表 | 本轮文件夹里的 `r<N>_report.py`、`ablate.py` 等 | 不动 |
| 模型的输入结构 | 历史位置数、历史怎么编码 | 改模型库 `src/flash_ansr/model/`，新结构放在模型配置开关后面（像已有的 `history_slots`），默认保持旧行为；本轮的 config 打开开关 | 一般不动 |
| core 确实缺一个通用功能 | 新的数据划分、新的训练接口 | 改 core，新行为放在默认关闭的配置开关后面；改完把每一轮的 `smoke.yaml` 都跑一遍 | 少量改，要验证 |

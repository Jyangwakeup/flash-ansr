# Unified history trainer: when to build it and how

**Status: deferred (assessed 2026-11-23).** The fixed loop with plug-ins in
`experiments/history_benefits/core/train.py` is enough for now. Rounds 3 and 4 each
supplied their own `batch_maker` / `validator` and left core untouched.

## Why not now

| Question | State on 2026-11-23 |
|---|---|
| Can a new round plug in without editing core? | Yes. r3 and r4 did not change core. |
| Is the trainer the bottleneck? | No. A 3M-parameter model trains 1,000–2,000 steps on CPU in a few hours. |
| Is the input structure settled? | No. It may become several histories, the formula without predictions, history fed through a different path, or an entirely different extra input. A trainer built now would have to guess the interface. |
| Does subclassing the original `Trainer` give these features for free? | No. It never passes `history_mask` to the model. Its streaming pipeline (`streaming.py`) builds no history samples. Its checkpoints keep no RNG state. Batch fetching, the model call and `wandb.init` are inlined in long methods. |

## Triggers: start as soon as any one holds

1. **A round needs a GPU.** `core/train.py` runs on CPU only, has no mixed precision, and
   builds samples one at a time in the main process, so on a GPU data generation becomes
   the bottleneck.
2. **A round needs a different `collate`**, i.e. samples become model inputs in a
   different way: several histories, or history no longer in data column 19.
   `core/train.py` imports `common.collate` directly, and `common.collate` hard-codes
   19 columns.
3. The model grows substantially, or a single round trains for more than a day.

## Design

**Location.** A new file, `src/flash_ansr/train/history_trainer.py`. Neither the original
`Trainer` nor `experiments/history_benefits/core/` changes, so earlier rounds keep
running on the code that produced their results.

**Independence from the input structure.** The trainer relies on one contract:

```python
batch = input_format.collate(samples, device)
# batch = {
#   "inputs": {...},          # passed as-is to model(**inputs); the trainer never looks inside
#   "labels": LongTensor,     # [b, T-1], <pad> where no loss is taken
#   "outlier_target": Tensor | None,
#   "point_mask": BoolTensor | None,
#   "kinds": list[str],       # per-sample kind, for per-kind CE
# }
logits = model(**batch["inputs"])[:, :-1]
```

To change the input structure, write a new `collate` and the matching model config
switch; the trainer does not change. The data width comes from
`model.encoder_max_n_variables + model.history_slots` and is never a literal.

**Sample source.** `source.batch(step, rng) -> list[sample]`, with a `kind` on each
sample. This is the signature of today's `batch_maker`, so `MixSampler` plugs in
unchanged. Two RNG modes:

- `sample_rng: sequential` keeps one RNG stream and saves it in checkpoints. This is
  today's behaviour.
- `sample_rng: per_step` makes each step's RNG a function of `(seed, step)` only. Worker
  processes can then build upcoming batches ahead of time, and the samples are the same
  for any number of workers.

**Reuse.**

- `build_optimizer`: `src/flash_ansr/train/optimizers.py`
- `pw_linear_schedule`: `src/flash_ansr/train/schedules.py`
- `_z_loss`: `src/flash_ansr/train/train.py`
- Checkpoint and RNG save/restore: `checkpoint` / `restore` in `core/train.py`, extended
  with the GradScaler and CUDA RNG state.

**New config keys.** `device`; `amp_dtype` (none / bf16 / fp16, with a GradScaler for
fp16); `compile`; `grad_clip` (default 2.0); loss weights (default 0.05 / 1e-4).

**Logging.** W&B logging goes through an injected logger object (anything with
`.log(dict, step)`), so `src/` never imports `experiments/`. Local `metrics.jsonl` and
`samples.jsonl` are still written.

## Acceptance

1. **Step-by-step equivalence.** On r3's `smoke.yaml`, group C, CPU, fp32: run 20 steps
   with the old `core/train.py` and 20 with the new trainer. Every step's loss and
   learning rate, and the final weights, must match exactly (`atol=0`).
2. **Unit tests** in `tests/test_train/test_history_trainer.py`:
   - A tiny model with `history_slots=1` trains for 2 steps.
   - 4 straight steps give the same weights as 2 steps, checkpoint, resume, 2 more steps.
   - In `per_step` mode, the samples are identical for any number of workers.
   - A `collate` without history runs through the same trainer unchanged.
3. **Once a GPU is available:** 50 steps in bf16 give a loss curve close to the CPU fp32 run.

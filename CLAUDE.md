# Rules for working in this repo

## Where experiment files go

Every experiment `<name>` is split across three places:

| What | Where |
|---|---|
| Code and scripts | `experiments/<name>/` |
| Raw run outputs (json, logs, checkpoints) | `runs/<name>/` |
| Analysis reports and write-ups | `experiment-results/<name>/` |

- Set all three paths before the first run. Never write outputs or reports into `experiments/`.
- Example: `experiments/T8vsT8-1/run.py` → `runs/T8vsT8-1/` → `experiment-results/T8vsT8-1/迁移对照.md`

## Log every experiment to Weights & Biases

Any new or changed script under `experiments/<name>/` that trains or evaluates must log to W&B through `experiments/exp_wandb.py`. Scripts that only summarize or plot existing results are exempt.

```python
sys.path.insert(0, str(ROOT / "experiments"))
from exp_wandb import Run
run = Run(config=cfg)              # group <name> and folder runs/<name>/ are inferred
run.log({"loss": loss}, step=step) # the metrics that matter: loss, solve rates, timings
run.summary({...}); run.finish()   # final numbers at the end
```

- Log progress while the script runs, not only at the end. For evaluations, log the running solve rate after each problem.
- Keep the local json/jsonl outputs as well. W&B is the viewer; the local files are the record.
- Never hard-code `mode` or `entity`. `exp_wandb` goes online when the user is logged in and offline otherwise.
- When reporting a new experiment, tell the user what shows up on W&B.

## How to write for this user

- Always reply in Chinese. Keep only code, file names and commands in their original form.
- Use plain words. Lead with a concrete example (a table with numbers, an analogy), then name the term if needed. Explain any technical term right after you use it.
- Write analysis reports in Chinese with tables. Give a one-sentence conclusion first. State plainly when the sample is too small to support a conclusion.
